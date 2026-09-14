# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.2153218495013599

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
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

## === cell 5
data_directory = Path('/kaggle/input/cassava-leaf-disease-classification/')
BASE_directory = os.path.join(data_directory)
!pwd # current working directory
print(BASE_directory)

## === cell 6
train_csv = pd.read_csv(os.path.join(BASE_directory, 'train.csv'))
train_csv.head()

## === cell 8
def get_mapping_dictionary() -> dict:
    num_to_disease_map = open(os.path.join(BASE_directory, 'label_num_to_disease_map.json'))
    num_to_disease_map_dict = json.load(num_to_disease_map)
    num_to_disease_map_dict = {int(key):num_to_disease_map_dict[key] for key in num_to_disease_map_dict}
    
    return num_to_disease_map_dict


def pair_indices_and_class_names(class_distribution_dict: dict) -> dict:
    num_to_disease_map_dict = get_mapping_dictionary()
    
    res_dict = {}
    
    for key in class_distribution_dict.keys():
        if key in num_to_disease_map_dict:
            new_key = num_to_disease_map_dict[key]
            res_dict[new_key] = class_distribution_dict[key]
    
    return res_dict

## === cell 9
def visualize_class_distributions_training_data(mapped_class_distribution_dict: dict):
    fig, ax = plt.subplots()

    classes = list(mapped_class_distribution_dict.keys())
    counts = list(mapped_class_distribution_dict.values())
    bar_colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green']

    ax.bar(classes, counts, color=bar_colors)

    for i, count in enumerate(counts):
        ax.text(classes[i], count + 10, str(count), ha='center', va='bottom')

    ax.set_xticks(classes)  # Use set_xticks to set the exact tick positions
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel('Number of training samples')
    ax.set_xlabel('Classes')
    ax.set_title('Number of samples in training data')

    plt.show()

## === cell 10
class_distribution_dict = {}
for i in range(len(train_csv)): 
    key = train_csv.loc[i, "label"]
    
    if key not in class_distribution_dict:
        class_distribution_dict[key] = 1
        
    else:
        class_distribution_dict[key] += 1
        
print(class_distribution_dict)
mapped_class_distribution_dict = pair_indices_and_class_names(class_distribution_dict)
classes_list = list(mapped_class_distribution_dict.keys())
print(mapped_class_distribution_dict)
visualize_class_distributions_training_data(mapped_class_distribution_dict)

## === cell 12
train_images_dir = os.path.join(BASE_directory, 'train_images')
mapping_dictionary = get_mapping_dictionary()

def plot_n_training_images(n_images: int, train_csv: pd.DataFrame, number_classes: int):
    for j in range(number_classes):
        filtered_df = train_csv[train_csv['label'] == j]
        random_rows = filtered_df.sample(n_images)

        fig, axes = plt.subplots(1, n_images, figsize=(30, 5))  

        for i in range(len(random_rows)):
            image_id = random_rows.iloc[i]['image_id']
            label = random_rows.iloc[i]['label']

            class_name = mapping_dictionary[label]

            img_path = os.path.join(train_images_dir, image_id)
            img = Image.open(img_path)

            axes[i].imshow(img)
            axes[i].set_title(f'{class_name} ({image_id})')
            axes[i].axis('off')

        plt.show()

## === cell 13
num_classes = len(mapped_class_distribution_dict)
plot_n_training_images(5, train_csv, 5)

## === cell 14
def plot_class_representative_images(n_images: int, train_csv: pd.DataFrame, number_classes: int):
    fig, axes = plt.subplots(1, number_classes, figsize=(30, 5))
    for j in range(number_classes):
        filtered_df = train_csv[train_csv['label'] == j]
        random_rows = filtered_df.sample(n_images)

        
        image_id = random_rows.iloc[0]['image_id']
        label = random_rows.iloc[0]['label']

        class_name = mapping_dictionary[label]

        img_path = os.path.join(train_images_dir, image_id)
        img = Image.open(img_path)

        axes[j].imshow(img)
        axes[j].set_title(f'{class_name} ({image_id})')
        axes[j].axis('off')

    plt.show()
        
plot_class_representative_images(1, train_csv, 5)

## === cell 16

class CassavaLeafDataset(Dataset):
    def __init__(self, image_ids: list, labels: list, image_dir: str, dimension=(224, 224), transform=None):
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
            img = img_transformed['image']

        return img, label

## === cell 18
resize_dimension = (224, 224) # ImageNet size
transform = A.Compose([
    A.Resize(width = resize_dimension[0], height = resize_dimension[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ToTensorV2(),
])

transform_test = A.Compose([
    A.Resize(width = resize_dimension[0], height = resize_dimension[1]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

## === cell 19
batch_size = 64 # number of samples for training
num_workers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_csv_file = os.path.join(BASE_directory, 'train.csv')
path_to_image_directory = os.path.join(BASE_directory, 'train_images')
print(path_to_image_directory)

train_csv = pd.read_csv(path_to_csv_file)

                                                    

    
x_train, x_val, y_train, y_val = train_test_split(train_csv['image_id'], 
                                                    train_csv['label'], 
                                                    test_size=0.1, 
                                                    random_state=1)

print(len(x_train))
print(len(x_val))


train_dataset = CassavaLeafDataset(
    image_ids=x_train.values,
    labels=y_train.values,
    image_dir=path_to_image_directory,
    dimension=resize_dimension,
    transform=transform
)


val_dataset = CassavaLeafDataset(
    image_ids=x_val.values,
    labels=y_val.values,
    image_dir=path_to_image_directory,
    dimension=resize_dimension,
    transform=transform
)



## === cell 21
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False
)







## === cell 23
print(f"Train set size: {len(train_loader.dataset)} images.")
print(f"Validation set size: {len(val_loader.dataset)} images.")



## === cell 25
classes = list(get_mapping_dictionary().values())

def imshow(img):
    img = img / 2 + 0.5     # denormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg))
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

## === cell 27
from torchvision import models

## === cell 28
dir(models) # available models and weights from torchvision

## === cell 29
print(torch.cuda.is_available()) # availability of GPU

## === cell 30
def create_actual_timestamp():
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S")
    
    return time_string

## === cell 31
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

## === cell 34
class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(ImageClassificationModel, self).__init__()
        
        self.pretrained_model = models.resnet50(pretrained=False)
        in_features = self.pretrained_model.fc.in_features
        
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        

    def forward(self, x):
        return self.pretrained_model(x)

## === cell 36
def train_classification_model(train_loader, val_loader, num_epochs, learning_rate, dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 5
    model_path = "/kaggle/input/resnet-50-model/resnet50.pth"
    weights = torch.load(model_path)
    model = ImageClassificationModel(num_classes, dropout).to(device)
    

    loss_function = nn.CrossEntropyLoss() # Cross entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) # Adam optimizer

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config["batch_size"])
    
    
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        avg_epoch_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            train_loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            epoch_loss += train_loss.item()
            
            metrics = {"train/train_loss": train_loss, 
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch}
            
                
            
        avg_epoch_loss = epoch_loss / n_steps_per_epoch
            

        model.eval()
        val_loss = 0
        with torch.no_grad():
            correct = 0
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.float().to(device), labels.to(device)
                
                outputs = model(images)
                val_loss += loss_function(outputs, labels) * labels.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(val_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:4f}, Accuracy: {accuracy:.4f}")
        
    print('Training finished.')
    return model

## === cell 39
trained_model = train_classification_model(train_loader, 
                                           val_loader, 
                                           num_epochs=config["num_epochs"], 
                                           learning_rate=config["learning_rate"], 
                                           dropout=config["dropout"])

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/547163249.py in <cell line: 0>()
----> 1 trained_model = train_classification_model(train_loader, 
      2                                            val_loader,
      3                                            num_epochs=config["num_epochs"],
      4                                            learning_rate=config["learning_rate"],
      5                                            dropout=config["dropout"])

/tmp/ipykernel_11/2948229544.py in train_classification_model(train_loader, val_loader, num_epochs, learning_rate, dropout)
      5     num_classes = 5
      6     model_path = "/kaggle/input/resnet-50-model/resnet50.pth"
----> 7     weights = torch.load(model_path)
      8     model = ImageClassificationModel(num_classes, dropout).to(device)
      9 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/resnet-50-model/resnet50.pth'

## === cell 44
trained_models_directory = Path("/kaggle/working/")
output_directory = os.path.join(trained_models_directory)
print(os.path.exists(output_directory))
torch.save(trained_model.state_dict(), os.path.join(output_directory, "model_resnet_50_dropout_0.pth"))

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2072871002.py in <cell line: 0>()
      2 output_directory = os.path.join(trained_models_directory)
      3 print(os.path.exists(output_directory))
----> 4 torch.save(trained_model.state_dict(), os.path.join(output_directory, "model_resnet_50_dropout_0.pth"))

NameError: name 'trained_model' is not defined

## === cell 46
def do_inference(img_path: str, model, transform, classes: list):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path)
    
    img_transformed = transform(image=np.array(img))
    img_tensor = img_transformed["image"]

    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)

    _, predicted_class = torch.max(output, 1)

    class_idx = predicted_class.item()
    
    return class_idx

## === cell 48
cassava_weights_path = os.path.join(output_directory, "model_resnet_50_dropout_0.pth")
test_images_path = os.path.join(BASE_directory, "test_images")

model = models.resnet50()
cassava_weights = torch.load(cassava_weights_path)

model.load_state_dict(cassava_weights, strict = False)

num_classes = len(classes_list)  # Number of cassava disease classes

in_features = model.fc.in_features
model.fc = nn.Linear(model.fc.in_features, num_classes)



    





## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3709503648.py in <cell line: 0>()
      7 # model = models.densenet121()
      8 model = models.resnet50()
----> 9 cassava_weights = torch.load(cassava_weights_path)
     10 
     11 # Update the pre-trained model with custom weights

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/model_resnet_50_dropout_0.pth'

## === cell 51
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
DATA_DIR_test = '/kaggle/input/cassava-leaf-disease-classification/test_images/'
X_Test = [name for name in (os.listdir(DATA_DIR_test))]
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.lbs = Labels
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        if "train" in self.dir:            
            return self.transform(x), self.lbs[index]            
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

Transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

testset = GetData(DATA_DIR_test, X_Test, None, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

s_ls = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader: 
        image = image.to(torch.float)
        
        logits = model(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            s_ls.append([fname[0], pred.item()])
            
sub = pd.DataFrame.from_records(s_ls, columns=['image_id', 'label'])

sub.head()

sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2062475438.py in <cell line: 0>()
     33 with torch.no_grad():
     34     model.eval()
---> 35     for image, fname in testloader:
     36         image = image.to(torch.float)
     37         #image = image.to('cuda')

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

IsADirectoryError: Caught IsADirectoryError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/2062475438.py", line 17, in __getitem__
    x = Image.open(os.path.join(self.dir, self.fnames[index]))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/cassava-leaf-disease-classification/test_images/test_images'


## === cell 52
top_class

## === cell 53
sub

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/35866194.py in <cell line: 0>()
----> 1 sub

NameError: name 'sub' is not defined
