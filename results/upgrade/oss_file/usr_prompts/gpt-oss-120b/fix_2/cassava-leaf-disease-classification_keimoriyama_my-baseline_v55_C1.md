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

No external packages required in the script and installed.

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

0.6128739800543971

# 6. Current score

0.74439

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.74439) has done: 'I fixed the missing pretrained weight files by using timm’s built‑in pretrained models, corrected the indentation error in `test_rate`, filtered out non‑image entries when loading the test set, and ensured the submission DataFrame is built from matching lists. These changes let the notebook run end‑to‑end, produce a valid `submission.csv`, and keep the original model architecture so the expected accuracy stays around the target score.'

# 9. Code solution

## === cell 0
try:
    import sys

    sys.path.append("../input/pretrined-models")
except Exception:
    pass



## === cell 1
import os
import pandas as pd
import timm



## === cell 2
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)



## === cell 3
df = pd.read_csv(path + "/train.csv")



## === cell 4
df.head()



## === cell 5
df["path"] = df["image_id"].map(lambda x: path + "/train_images/" + x)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1).reset_index(drop=True)



## === cell 6
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 7
train_df.label.value_counts().plot(kind="bar")



## === cell 8
valid_df.label.value_counts().plot(kind="bar")



## === cell 9
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()



## === cell 10
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()



## === cell 11
from PIL import Image, ImageDraw

im = Image.open(train_df["path"][0])



## === cell 12
im



## === cell 13
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold

import matplotlib.image as img




## === cell 14
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




## === cell 15
import random




## === cell 16
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




## === cell 17
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



## === cell 18
dataset = CassavaDataset(train_df, train_transform)



## === cell 19
import torch.nn as nn



## === cell 20
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 21
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)



## === cell 22
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)



## === cell 23
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 24
def calc_correction(model, df):
    model.eval()
    path = df["path"]
    label = df["label"]
    count = 0
    pred_list = [0, 0, 0, 0, 0]
    for i in range(len(path)):
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




## === cell 25
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses, label="train")
    valid_loss = plt.plot(y, valid_losses, label="valid")
    plt.title(title)
    plt.ylabel("loss")
    plt.legend()
    plt.show()




## === cell 26
import time


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_model = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        print("fold:", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)

        optimizer = type(optimizer)(model.parameters(), lr=1e-4, weight_decay=0.0001)
        scheduler = type(scheduler)(optimizer, step_size=2, gamma=0.1)

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            train_loss = 0.0
            model.train()
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    valid_loss += loss.item() * data.size(0)
                    preds = output.argmax(1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
            valid_loss /= len(valid_loader.dataset)
            valid_losses.append(valid_loss)

            acc = correct / total
            scheduler.step()
            print(
                f"Time: {time.time() - epoch_start_time:.3f}s Epoch: {ep} "
                f"TrainLoss: {train_loss:.4f} ValidLoss: {valid_loss:.4f} Acc: {acc:.4f}"
            )

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = type(model)(*model.children())
                best_model.load_state_dict(model.state_dict())

    torch.save(best_model.state_dict(), model_title)
    return best_model, train_losses, valid_losses




## === cell 27
def train_models():
    res_model_path = "./res_model.pth"
    best_resnet, train_losses, valid_losses = train_model(
        resNet,
        dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        res_model_path,
    )
    print("ResNet validation accuracy:", calc_correction(best_resnet, valid_df)[0])
    plot_losses(epoch, "ResNet losses", train_losses, valid_losses)

    ef_model_path = "./ef_model.pth"
    best_ef, train_losses, valid_losses = train_model(
        ef_model,
        dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        ef_model_path,
    )
    print("EfficientNet validation accuracy:", calc_correction(best_ef, valid_df)[0])
    plot_losses(epoch, "EfficientNet losses", train_losses, valid_losses)

    return best_resnet, best_ef




## === cell 28
resNet, ef_model = train_models()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/847205736.py in <cell line: 0>()
      1 # Train both models and obtain the fine‑tuned versions
----> 2 resNet, ef_model = train_models()
      3 

/tmp/ipykernel_55/945969571.py in train_models()
      2     # Train ResNet
      3     res_model_path = "./res_model.pth"
----> 4     best_resnet, train_losses, valid_losses = train_model(
      5         resNet,
      6         dataset,

/tmp/ipykernel_55/2760259089.py in train_model(model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title)
     63             if valid_loss < best_loss:
     64                 best_loss = valid_loss
---> 65                 best_model = type(model)(*model.children())
     66                 best_model.load_state_dict(model.state_dict())
     67 

/usr/local/lib/python3.11/dist-packages/timm/models/resnet.py in __init__(self, block, layers, num_classes, in_chans, output_stride, global_pool, cardinality, base_width, stem_width, stem_type, replace_stem_pool, block_reduce_first, down_kernel_size, avg_down, channels, act_layer, norm_layer, aa_layer, drop_rate, drop_path_rate, drop_block_rate, zero_init_last, block_args)
    478         super(ResNet, self).__init__()
    479         block_args = block_args or dict()
--> 480         assert output_stride in (8, 16, 32)
    481         self.num_classes = num_classes
    482         self.drop_rate = drop_rate

AssertionError: 

## === cell 30
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.3 * x1 + 0.7 * x2

    def test(self, x, rate):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return rate * x1 + (1 - rate) * x2




## === cell 31
classifier = CassaveClassifier(resNet, ef_model).to(device)




## === cell 32
def test_rate():
    for rate in range(1, 10):
        classifier.eval()
        path = valid_df["path"]
        label = valid_df["label"]
        count = 0
        pred_list = [0, 0, 0, 0, 0]
        for i in range(len(path)):
            image_path = path[i]
            image_label = label[i]
            image = Image.open(image_path)
            image = valid_transform(image)
            image = image.unsqueeze(0).to(device)
            pred = classifier.test(image, rate / 10.0).argmax(1).item()
            pred_list[pred] += 1
            if pred == image_label:
                count += 1
        percent = count / len(path)
        print("rate:", rate / 10.0, "percent:", percent)




## === cell 33
test_path = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 34
image_path_list = []
image_id_list = []
for entry in os.listdir(test_path):
    full_path = os.path.join(test_path, entry)
    if os.path.isfile(full_path):  # skip nested directories
        image_id_list.append(entry)
        image_path_list.append(full_path)



## === cell 35
pred = []
classifier.eval()
for p in image_path_list:
    image = Image.open(p)
    image = valid_transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        predict = classifier(image).argmax(1).item()
    pred.append(predict)



## === cell 36
sub = pd.DataFrame({"image_id": image_id_list, "label": pred})



## === cell 37
sub.head()



## === cell 38
sub.to_csv("submission.csv", index=False)
