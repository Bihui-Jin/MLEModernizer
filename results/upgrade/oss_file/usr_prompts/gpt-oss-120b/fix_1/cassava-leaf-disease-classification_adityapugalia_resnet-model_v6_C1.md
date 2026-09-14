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

3.13

# 3. Installed packages

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

0.8112722877002115

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import pandas as pd
import numpy as np 
import os
import json 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt

## === cell 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## === cell 5
base_path = '/kaggle/input/cassava-leaf-disease-classification/'

train_path = '/kaggle/input/cassava-leaf-disease-classification/train_images/'

test_path = '/kaggle/input/cassava-leaf-disease-classification/test_images/'

with open(base_path+'label_num_to_disease_map.json') as f :
    mapping = json.loads(f.read())
    mapping = {int(k): v for k, v in mapping.items()}
mapping

## === cell 6
train_data = pd.read_csv(base_path + 'train.csv')
train_data.head()

## === cell 7
from collections import defaultdict
mapping_count = {0 : 0, 1: 0, 2: 0, 3: 0, 4: 0}
total_img = 0
for i in range(len(train_data)):
    mapping_count[train_data.label[i]] += 1
    total_img += 1
mapping_count = pd.DataFrame(mapping_count.items(), columns= ['label', 'image_count'])
mapping_count

## === cell 9
for i in range(5):
    image_path = train_path + train_data.image_id[i]
    with Image.open(image_path) as img:
        print(f"Image: {train_data.image_id[i]} | Dimensions: {img.size}")

## === cell 10
import multiprocessing

img_height, img_width = 800, 600  # Resize all images to 800x600 pixels
batch_size = 32  # Number of images to process in a batch

image_dir = train_path  # Assuming train_path is defined elsewhere


data_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # Resize images to the desired size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

class ImageDataset(Dataset):
    def __init__(self, csv_data, image_dir, transform=None):
        self.image_paths = csv_data['image_id'].values  # Paths to images
        self.labels = csv_data['label'].values  # Labels
        self.image_dir = image_dir  # Directory where images are stored
        self.transform = transform  # Transformation to apply to each image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])  # Full path to the image
        image = Image.open(image_path).convert("RGB")  # Open the image and convert to RGB
        label = self.labels[idx]  # Get the corresponding label
        
        if self.transform:
            image = self.transform(image)  # Apply the transformations

        return image, label  # Return the image and the label

img_dataset = ImageDataset(train_data, image_dir, transform=data_transforms)

def split_dataset(dataset, train_size=0.8):
    """
    Splits a dataset into train and validation sets.
    
    Args:
    - dataset: The dataset to split.
    - train_size: The proportion of the data to use for training.
    
    Returns:
    - train_dataset: The training dataset.
    - val_dataset: The validation dataset.
    """
    train_length = int(len(dataset) * train_size)
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    return train_dataset, val_dataset


total_images = len(img_dataset)

train_dataset, val_dataset = split_dataset(img_dataset, train_size=0.8)

print('computing class weights')
class_weights = [mapping_count['image_count'][i]/total_img for i in range(5)]

sample_weights_all = [class_weights[label] for label in img_dataset.labels]

train_indices = train_dataset.indices
train_sample_weights = [sample_weights_all[i] for i in train_indices]
print("created sample weights")

sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
print("created sampler")

num_workers = multiprocessing.cpu_count()   
print("numworkers: ", num_workers)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = num_workers, prefetch_factor=2)
print('created train loader')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, prefetch_factor=2)
print('created valid loader')

print(f'Total images: {total_images}')
print(f'Training images: {len(train_dataset)}')
print(f'Validation images: {len(val_dataset)}')

## === cell 11
base_model = models.resnet50(weights= None)

base_model.load_state_dict(torch.load("/kaggle/input/resnet50-weights/resnet50_weights.pth"))

for param in base_model.parameters():
    param.requires_grad = False

num_features = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Linear(num_features, 384),  # Fully connected layer with 384 neurons
    nn.ReLU(),                     # ReLU activation
    nn.Linear(384, 5),             # Output layer with 5 classes
)

base_model = base_model.to(device)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)  # Only train the FC layer

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0  # Track the best validation accuracy
    
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/kaggle/working/ResModel.pt")
            print("Model saved with Validation Accuracy: {:.4f}".format(best_val_acc))
    print("Training complete. Best Validation Accuracy: {:.4f}".format(best_val_acc))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/4256387735.py in <cell line: 0>()
      2 base_model = models.resnet50(weights= None)
      3 
----> 4 base_model.load_state_dict(torch.load("/kaggle/input/resnet50-weights/resnet50_weights.pth"))
      5 
      6 # Freeze the base model layers

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/resnet50-weights/resnet50_weights.pth'

## === cell 13
base_model.load_state_dict(torch.load("/kaggle/input/res-model/ResModel.pt"))

test_df = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
test_dataset = ImageDataset(test_df, "/kaggle/input/cassava-leaf-disease-classification/test_images", transform= data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

predictions = []
with torch.no_grad():  # Disable gradient computation for inference
    for images, _ in test_loader:
        images = images.to(device)
        outputs = base_model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())  # Store predictions

predictions = np.array(predictions)
    

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/4087226704.py in <cell line: 0>()
----> 1 base_model.load_state_dict(torch.load("/kaggle/input/res-model/ResModel.pt"))
      2 
      3 # 4. Predict on test data
      4 test_df = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
      5 test_dataset = ImageDataset(test_df, "/kaggle/input/cassava-leaf-disease-classification/test_images", transform= data_transforms)

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/res-model/ResModel.pt'

## === cell 14
print(len(predictions))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/155380672.py in <cell line: 0>()
----> 1 print(len(predictions))

NameError: name 'predictions' is not defined

## === cell 15
test_df['label'] = predictions
test_df.to_csv('submission.csv', index=False)
print("Submission file created!")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3533822622.py in <cell line: 0>()
      1 # 5. Create submission file
----> 2 test_df['label'] = predictions
      3 test_df.to_csv('submission.csv', index=False)
      4 print("Submission file created!")

NameError: name 'predictions' is not defined
