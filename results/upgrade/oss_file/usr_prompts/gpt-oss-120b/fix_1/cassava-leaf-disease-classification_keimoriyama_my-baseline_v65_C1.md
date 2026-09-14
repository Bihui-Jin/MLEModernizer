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

# 5. Target score

0.4584466606225446

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip install --no-deps ../input/pretrined-models/timm-0.3.3-py3-none-any.whl

## === cell 2
import os
import pandas as pd
import timm
from PIL import Image, ImageDraw, ImageChops
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

## === cell 3
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)

## === cell 4
df = pd.read_csv(path + "/train.csv")

## === cell 5
df.head()

## === cell 6
df["path"] = df["image_id"].map(lambda x: path + "/train_images/" + x)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1).reset_index(drop=True)

## === cell 7
df.head()

## === cell 8
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)

## === cell 9
train_df.label.value_counts().plot(kind="bar")

## === cell 10
valid_df.label.value_counts().plot(kind="bar")

## === cell 11
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()

## === cell 12
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()

## === cell 13
im = Image.open(train_df["path"][0])

## === cell 14
im

## === cell 15
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold

import matplotlib.image as img

## === cell 17
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df["path"])

    def __getitem__(self, index):
        path = self.df["path"][index]
        label = self.df["label"][index]
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label

## === cell 18
import random

## === cell 19
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for i in range(10):
                start_width.append(random.randrange(0, width - self.mask_size))
                start_height.append(random.randrange(0, height - self.mask_size))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image

## === cell 20
image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose(
    [  # 大きさの変更
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
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

## === cell 21
dataset = CassavaDataset(train_df, train_transform)

## === cell 23
import json

path = '../input/cassava-leaf-disease-classification/label_num_to_disease_map.json'

with open(path, mode = 'r') as f:
    label_to_name = json.load(f)

## === cell 24
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

## === cell 25
unnorm = Unnormalize(mean, std)

## === cell 26
def display_img(img, unnorm = None, label = None):
    if unnorm != None:
        img = unnorm(img)
        
    plt.imshow(img.permute(1, 2, 0))
    
    if label != None:
        plt.title(label_to_name[str(label)])

## === cell 27
def display_batch(batch, unnorm = None):
    imgs, labels = batch
    
    if unnorm:
        unnorm_imgs = []
        for img in imgs:
            unnorm_imgs.append(unnorm(img))
        imgs = unnorm_imgs
        
    ig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))

## === cell 28
tensor, label = dataset[0]
display_img(tensor, unnorm, label)

## === cell 29
loader = DataLoader(dataset, 16, shuffle = True)
display_batch(next(iter(loader)))

## === cell 30
import torch
import torch.nn as nn
import torch.nn.functional as F

## === cell 31
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## === cell 32
resNet = timm.create_model("resnet50", pretrained=False)
resNet.load_state_dict(torch.load("../input/pretrined-models/models/models/pretrained_resNet.pth"))
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/3564158059.py in <cell line: 0>()
      1 resNet = timm.create_model("resnet50", pretrained=False)
----> 2 resNet.load_state_dict(torch.load("../input/pretrined-models/models/models/pretrained_resNet.pth"))
      3 resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
      4 resNet = resNet.to(device)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/pretrined-models/models/models/pretrained_resNet.pth'

## === cell 33
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)

ef_model.load_state_dict(torch.load("../input/pretrined-models/models/models/pretrained_ef_model.pth"))

ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/1153112508.py in <cell line: 0>()
      1 ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)
      2 
----> 3 ef_model.load_state_dict(torch.load("../input/pretrined-models/models/models/pretrained_ef_model.pth"))
      4 
      5 ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/pretrined-models/models/models/pretrained_ef_model.pth'

## === cell 34
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(resNet_optimizer, step_size=2, gamma=0.1)
criterion=nn.CrossEntropyLoss()

## === cell 35
def calc_correction(model, df):
    model.eval()
    path = df["path"]
    label = df["label"]
    count = 0
    pred_list = [0, 0, 0, 0, 0]
    for i in tqdm(range(len(path))):
        image_path = path[i]
        image_label = label[i]
        image = Image.open(image_path)
        image = valid_transform(image)
        image = image.unsqueeze(0).to(device)
        model = model.to(device)
        pred = model(image).argmax(1).item()
        pred_list[pred] += 1
        if pred == image_label:
            count += 1
    percent = count / len(path)
    return percent, pred_list

## === cell 36
from matplotlib import pyplot as plt
def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend(
        (train_loss[0], valid_loss[0]), ("train loss", "valid loss"),
    )
    plt.show()


## === cell 37
import time
def train_model(model, dataset, batch_size, optimizer, criterion, scheduler,  epoch, model_title):
    best_model = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits = 4)
    
    for fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        print("fold: ", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 4)
        valid_dataset = Subset(dataset, valid_index)
        valid_loader= DataLoader(valid_dataset, batch_size, shuffle = False)
        
        for epoch in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0
            valid_loss = 0

            model.train()
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*len(data)
            train_loss = train_loss/len(train_loader.sampler)
            train_losses.append(train_loss)

            model.eval()
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)

                with torch.no_grad():
                    output = model(data)
                    pred = (output.argmax(1) == target)
                    acc.append(sum(pred)/len(pred))

                    loss = criterion(output, target)

                    valid_loss += loss.item()*len(data)
                    
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = model
                    
            scheduler.step()

            collection = sum(acc)/len(acc)
            valid_loss = valid_loss/len(valid_loader.sampler)
            valid_losses.append(valid_loss)
            print('Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}'
                  .format(time.time() - epoch_start_time, epoch, train_loss, valid_loss, collection))
            num_collection = []
    torch.save(best_model.state_dict(), model_title)
    
    return model, train_losses, valid_losses

## === cell 38
def train_models(resNet, ef_model):
    model_title = "./res_model.pth"
    resNet, train_losses, valid_losses = train_model(resNet, dataset, batch_size, resNet_optimizer, criterion,resNet_scheduler, epoch, model_title)
    print(calc_correction(resNet, valid_df))
    title = "resNet losses"
    plot_losses(epoch, title, train_losses, valid_losses)
    model_title = "./ef_model.pth"
    ef_model, train_losses, valid_losses = train_model(ef_model, dataset, batch_size, ef_optimizer, criterion, ef_scheduler, epoch, model_title)
    print(calc_correction(ef_model, valid_df))
    title = "ef losses"
    plot_losses(epoch, title, train_losses, valid_losses)

## === cell 40
ef_model.load_state_dict(torch.load("../input/pretrined-models/models/models/ef_model.pth", map_location = device))
resNet.load_state_dict(torch.load("../input/pretrined-models/models/models/res_model.pth", map_location = device))

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/372803702.py in <cell line: 0>()
----> 1 ef_model.load_state_dict(torch.load("../input/pretrined-models/models/models/ef_model.pth", map_location = device))
      2 resNet.load_state_dict(torch.load("../input/pretrined-models/models/models/res_model.pth", map_location = device))

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/pretrined-models/models/models/ef_model.pth'

## === cell 41
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model
    
    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return (0.5 * x1 + 0.5 * x2)

    def test(self, x, rate):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        p = rate * x1 + (1 - rate) * x2
        return p

## === cell 42
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)

## === cell 43
def test_rate():
    for rate in range(1, 10):
        classifier.eval()
        path = valid_df["path"]
        label = valid_df["label"]
        count = 0
        pred_list = [0, 0, 0, 0, 0]
        for i in tqdm(range(len(path))):
            image_path = path[i]
            image_label = label[i]
            image = Image.open(image_path)
            image = valid_transform(image)
            image = image.unsqueeze(0).to(device)
            pred = classifier.test(image, rate/10).argmax(1).item()
            pred_list[pred] += 1
            if pred == image_label:
                count += 1
        percent = count / len(path)
        print("rate: ", rate/10)
        print("percent: ", percent)

## === cell 46
path = "../input/cassava-leaf-disease-classification/test_images/"

## === cell 47
image_path = []
image_id = []
for i in os.listdir(path):
    image_id.append(str(i))
    image_path.append(path + str(i))

## === cell 48
pred = []
for path in image_path:
    image = Image.open(path)
    image = valid_transform(image)
    image = image.unsqueeze(0).to(device)
    predict = resNet(image).argmax(1).item()
    pred.append(predict)

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_10/31650492.py in <cell line: 0>()
      1 pred = []
      2 for path in image_path:
----> 3     image = Image.open(path)
      4     image = valid_transform(image)
      5     image = image.unsqueeze(0).to(device)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/cassava-leaf-disease-classification/test_images/test_images'

## === cell 49
pred

## === cell 50
sub = pd.DataFrame({"image_id": image_id, "label": pred})

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/3219014970.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"image_id": image_id, "label": pred})

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 51
sub

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/35866194.py in <cell line: 0>()
----> 1 sub

NameError: name 'sub' is not defined

## === cell 52
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/352017882.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)

NameError: name 'sub' is not defined
