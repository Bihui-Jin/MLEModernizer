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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
wandb==0.21.0

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
import numpy as np
from pathlib import Path
import os
import math
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image

import wandb

from sklearn.model_selection import train_test_split

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from albumentations.pytorch import ToTensorV2
import albumentations as A

from datetime import datetime

torch.backends.cudnn.benchmark = True




## === cell 1
data_directory = Path("/kaggle/input/cassava-leaf-disease-classification/")
BASE_directory = os.path.join(data_directory)
print("Base directory:", BASE_directory)




## === cell 2
train_csv = pd.read_csv(os.path.join(BASE_directory, "train.csv"))
train_csv.head()




## === cell 3
def get_mapping_dictionary() -> dict:
    num_to_disease_map = open(
        os.path.join(BASE_directory, "label_num_to_disease_map.json")
    )
    num_to_disease_map_dict = json.load(num_to_disease_map)
    num_to_disease_map_dict = {
        int(key): num_to_disease_map_dict[key] for key in num_to_disease_map_dict
    }
    return num_to_disease_map_dict


def pair_indices_and_class_names(class_distribution_dict: dict) -> dict:
    num_to_disease_map_dict = get_mapping_dictionary()
    res_dict = {}
    for key in class_distribution_dict.keys():
        if key in num_to_disease_map_dict:
            new_key = num_to_disease_map_dict[key]
            res_dict[new_key] = class_distribution_dict[key]
    return res_dict




## === cell 4
def visualize_class_distributions_training_data(mapped_class_distribution_dict: dict):
    fig, ax = plt.subplots()
    classes = list(mapped_class_distribution_dict.keys())
    counts = list(mapped_class_distribution_dict.values())
    bar_colors = ["tab:red", "tab:blue", "tab:orange", "tab:purple", "tab:green"]
    ax.bar(classes, counts, color=bar_colors)
    for i, count in enumerate(counts):
        ax.text(classes[i], count + 10, str(count), ha="center", va="bottom")
    ax.set_xticks(classes)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel("Number of training samples")
    ax.set_xlabel("Classes")
    ax.set_title("Number of samples in training data")
    plt.show()




## === cell 5
class_distribution_dict = {}
for i in range(len(train_csv)):
    key = train_csv.loc[i, "label"]
    if key not in class_distribution_dict:
        class_distribution_dict[key] = 1
    else:
        class_distribution_dict[key] += 1
print("Raw class counts:", class_distribution_dict)

mapped_class_distribution_dict = pair_indices_and_class_names(class_distribution_dict)
print("Mapped class counts:", mapped_class_distribution_dict)
visualize_class_distributions_training_data(mapped_class_distribution_dict)




## === cell 6
train_images_dir = os.path.join(BASE_directory, "train_images")
mapping_dictionary = get_mapping_dictionary()


def plot_n_training_images(n_images: int, train_csv: pd.DataFrame, number_classes: int):
    for j in range(number_classes):
        filtered_df = train_csv[train_csv["label"] == j]
        random_rows = filtered_df.sample(n_images)
        fig, axes = plt.subplots(1, n_images, figsize=(30, 5))
        for i in range(len(random_rows)):
            image_id = random_rows.iloc[i]["image_id"]
            label = random_rows.iloc[i]["label"]
            class_name = mapping_dictionary[label]
            img_path = os.path.join(train_images_dir, image_id)
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"{class_name} ({image_id})")
            axes[i].axis("off")
        plt.show()




## === cell 7
num_classes = len(mapped_class_distribution_dict)
plot_n_training_images(5, train_csv, num_classes)




## === cell 8
def plot_class_representative_images(
    n_images: int, train_csv: pd.DataFrame, number_classes: int
):
    fig, axes = plt.subplots(1, number_classes, figsize=(30, 5))
    for j in range(number_classes):
        filtered_df = train_csv[train_csv["label"] == j]
        random_rows = filtered_df.sample(n_images)
        image_id = random_rows.iloc[0]["image_id"]
        label = random_rows.iloc[0]["label"]
        class_name = mapping_dictionary[label]
        img_path = os.path.join(train_images_dir, image_id)
        img = Image.open(img_path)
        axes[j].imshow(img)
        axes[j].set_title(f"{class_name} ({image_id})")
        axes[j].axis("off")
    plt.show()


plot_class_representative_images(1, train_csv, num_classes)




## === cell 9
class CassavaLeafDataset(Dataset):
    def __init__(
        self,
        image_ids: list,
        labels: list,
        image_dir: str,
        dimension=(224, 224),
        transform=None,
    ):
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.dimension = dimension
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_ids[idx])
        img = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            img_transformed = self.transform(image=np.array(img))
            img = img_transformed["image"]
        return img, label




## === cell 10
resize_dimension = (224, 224)  # ImageNet size
transform = A.Compose(
    [
        A.Resize(width=resize_dimension[0], height=resize_dimension[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

transform_test = A.Compose(
    [
        A.Resize(width=resize_dimension[0], height=resize_dimension[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)




## === cell 11
batch_size = 64
num_workers = min(4, os.cpu_count() or 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_csv_file = os.path.join(BASE_directory, "train.csv")
path_to_image_directory = os.path.join(BASE_directory, "train_images")
print("Image directory:", path_to_image_directory)

train_csv = pd.read_csv(path_to_csv_file)

x_train, x_val, y_train, y_val = train_test_split(
    train_csv["image_id"], train_csv["label"], test_size=0.1, random_state=1
)

print("Train split size:", len(x_train))
print("Validation split size:", len(x_val))

train_dataset = CassavaLeafDataset(
    image_ids=x_train.values,
    labels=y_train.values,
    image_dir=path_to_image_directory,
    dimension=resize_dimension,
    transform=transform,
)

val_dataset = CassavaLeafDataset(
    image_ids=x_val.values,
    labels=y_val.values,
    image_dir=path_to_image_directory,
    dimension=resize_dimension,
    transform=transform,
)




## === cell 12
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=False,
    persistent_workers=True,
)




## === cell 13
print(f"Train set size: {len(train_loader.dataset)} images.")
print(f"Validation set size: {len(val_loader.dataset)} images.")




## === cell 14
classes = list(get_mapping_dictionary().values())


def imshow(img):
    img = img / 2 + 0.5  # denormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(" ".join("%5s" % classes[labels[j]] for j in range(min(batch_size, len(labels)))))




## === cell 15
from torchvision import models




## === cell 16
print("Available model names count:", len(dir(models)))




## === cell 17
print("CUDA available:", torch.cuda.is_available())




## === cell 18
def create_actual_timestamp():
    current_time = datetime.now()
    return current_time.strftime("%Y%m%d%H%M%S")




## === cell 19
config = {
    "learning_rate": 1e-3,
    "batch_size": batch_size,
    "architecture": "",
    "dataset": "Cassava leaf disease",
    "num_epochs": 30,
    "dropout": 0,
    "device": device,
    "timestamp": create_actual_timestamp(),
}




## === cell 20
class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(ImageClassificationModel, self).__init__()
        self.pretrained_model = models.resnet50(pretrained=False)
        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.pretrained_model(x)




## === cell 21
def train_classification_model(
    train_loader, val_loader, num_epochs, learning_rate, dropout
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    model = ImageClassificationModel(num_classes, dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config["batch_size"])
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            train_loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        avg_epoch_loss = epoch_loss / n_steps_per_epoch
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += loss_function(outputs, labels) * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(val_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
    print("Training finished.")
    return model




## === cell 22
trained_model = train_classification_model(
    train_loader,
    val_loader,
    num_epochs=config["num_epochs"],
    learning_rate=config["learning_rate"],
    dropout=config["dropout"],
)




## === cell 23
trained_models_directory = Path("/kaggle/working/")
output_directory = trained_models_directory
os.makedirs(output_directory, exist_ok=True)
model_save_path = os.path.join(output_directory, "model_resnet_50_dropout_0.pth")
torch.save(trained_model.state_dict(), model_save_path)
print("Model saved to:", model_save_path)




## === cell 24
def do_inference(img_path: str, model, transform):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path).convert("RGB")
    img_transformed = transform(image=np.array(img))
    img_tensor = img_transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    _, pred = torch.max(output, 1)
    return pred.item()




## === cell 25
num_classes = len(classes)  # should be 5
inference_model = ImageClassificationModel(num_classes, dropout_prob=0).to(
    config["device"]
)
inference_model.load_state_dict(
    torch.load(model_save_path, map_location=config["device"])
)
inference_model.eval()




## === cell 26
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

DATA_DIR_test = os.path.join(BASE_directory, "test_images")
test_filenames = [
    f
    for f in os.listdir(DATA_DIR_test)
    if os.path.isfile(os.path.join(DATA_DIR_test, f))
]


class GetData(Dataset):
    def __init__(self, dir_path, fnames, transform):
        self.dir = dir_path
        self.fnames = fnames
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.fnames[index])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, self.fnames[index]


Transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

testset = GetData(DATA_DIR_test, test_filenames, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

predictions = []
with torch.no_grad():
    for images, fnames in testloader:
        images = images.to(config["device"])
        logits = inference_model(images)
        pred_label = torch.argmax(logits, dim=1).item()
        predictions.append([fnames[0], pred_label])

sub = pd.DataFrame(predictions, columns=["image_id", "label"])
sub.head()
sub.to_csv(os.path.join(output_directory, "submission.csv"), index=False)
print("Submission saved to:", os.path.join(output_directory, "submission.csv"))
