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
import os, subprocess, sys

wheel_path = "../input/pretrined-models/timm-0.3.3-py3-none-any.whl"
if os.path.exists(wheel_path):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", wheel_path]
    )
else:
    print(
        f"[WARN] Wheel not found at {wheel_path}; using the preinstalled timm package."
    )



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
pass




## === cell 15
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

        return image, int(label)




## === cell 16
import random




## === cell 17
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




## === cell 18
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



## === cell 19
pass



## === cell 20
dataset = CassavaDataset(train_df, train_transform)



## === cell 21
import torch
import torch.nn as nn
import torch.nn.functional as F



## === cell 22
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 23
resNet = timm.create_model("resnet50", pretrained=False)

resnet_w_path = "../input/pretrined-models/models/models/pretrained_resNet.pth"
if os.path.exists(resnet_w_path):
    resNet.load_state_dict(torch.load(resnet_w_path, map_location="cpu"))
else:
    print(f"[WARN] Missing {resnet_w_path}; using timm pretrained weights instead.")
    resNet = timm.create_model("resnet50", pretrained=True)

resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)



## === cell 24
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)

ef_w_path = "../input/pretrined-models/models/models/pretrained_ef_model.pth"
if os.path.exists(ef_w_path):
    ef_model.load_state_dict(torch.load(ef_w_path, map_location="cpu"))
else:
    print(f"[WARN] Missing {ef_w_path}; using timm pretrained weights instead.")
    ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)

ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)



## === cell 25
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 26
def calc_correction(model, df):
    model.eval()
    paths = df["path"].values
    labels = df["label"].astype(int).values
    count = 0
    pred_list = [0] * num_classes
    with torch.no_grad():
        for image_path, image_label in zip(paths, labels):
            image = Image.open(image_path).convert("RGB")
            image = valid_transform(image)
            image = image.unsqueeze(0).to(device)
            pred = model(image).argmax(1).item()
            if 0 <= pred < num_classes:
                pred_list[pred] += 1
            if pred == int(image_label):
                count += 1
    percent = count / len(paths)
    return percent, pred_list




## === cell 27
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend(
        (train_loss[0], valid_loss[0]),
        ("train loss", "valid loss"),
    )
    plt.show()




## === cell 28
import time
import copy


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=5)

    for fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        print("fold: ", fold)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler.step_size, gamma=scheduler.gamma
        )

        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)

        for epoch_i in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data)
            train_loss = train_loss / len(train_loader.sampler)
            train_losses.append(train_loss)

            model.eval()
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)

                with torch.no_grad():
                    output = model(data)
                    pred = output.argmax(1) == target
                    acc.append(sum(pred) / len(pred))

                    loss = criterion(output, target)
                    valid_loss += loss.item() * len(data)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())

            scheduler.step()

            collection = sum(acc) / len(acc)
            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_losses.append(valid_loss)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    epoch_i,
                    train_loss,
                    valid_loss,
                    collection,
                )
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_title)

    return model, train_losses, valid_losses




## === cell 29
model_title = "./res_model.pth"



## === cell 30
title = "resNet losses"



## === cell 31
model_title = "./ef_model.pth"



## === cell 32
title = "ef losses"



## === cell 33
fine_ef_path = "../input/pretrined-models/models/models/fine_tuned_ef_model.pth"
res_model_path = "../input/pretrined-models/models/models/res_model.pth"

need_train_res = not os.path.exists(res_model_path)
need_train_ef = not os.path.exists(fine_ef_path)

if not need_train_ef:
    ef_model.load_state_dict(torch.load(fine_ef_path, map_location="cpu"))
else:
    print(
        f"[WARN] Missing {fine_ef_path}; will fine-tune ef_model with the existing training loop."
    )

if not need_train_res:
    resNet.load_state_dict(torch.load(res_model_path, map_location="cpu"))
else:
    print(
        f"[WARN] Missing {res_model_path}; will fine-tune resNet with the existing training loop."
    )

ef_model = ef_model.to(device)
resNet = resNet.to(device)



## === cell 34
if need_train_res:
    resNet, res_train_losses, res_valid_losses = train_model(
        resNet,
        dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        "./res_model.pth",
    )

if need_train_ef:
    ef_model, ef_train_losses, ef_valid_losses = train_model(
        ef_model,
        dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        "./ef_model.pth",
    )




## === cell 35
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
        p = rate * x1 + (1 - rate) * x2
        return p




## === cell 36
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)



## === cell 37
"""
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
        model = classifier.to(device)
        pred = classifier.test(image, rate/10).argmax(1).item()
        pred_list[pred] += 1
        if pred == image_label:
            count += 1
    percent = count / len(path)
    print("rate: ", rate/10)
    print("percent: ", percent)
"""


## === cell 38
calc_correction(classifier, valid_df)



## === cell 39
path = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 40
valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
image_id = []
image_path = []
for fn in sorted(os.listdir(path)):
    full = os.path.join(path, fn)
    if os.path.isfile(full) and os.path.splitext(fn.lower())[1] in valid_ext:
        image_id.append(fn)
        image_path.append(full)

len(image_id), image_id[:5]



## === cell 41
sample_sub = pd.read_csv(
    os.path.join(
        "../input/cassava-leaf-disease-classification", "sample_submission.csv"
    )
)
sample_ids = sample_sub["image_id"].tolist()
id_to_path = {os.path.basename(p): p for p in image_path}
image_id = sample_ids
image_path = [id_to_path[i] for i in image_id]

len(image_id), image_id[:5]



## === cell 42
classifier.eval()
pred = []
with torch.no_grad():
    for pth in image_path:
        image = Image.open(pth).convert("RGB")
        image = valid_transform(image)
        image = image.unsqueeze(0).to(device)
        predict = classifier(image).argmax(1).item()
        pred.append(int(predict))

len(pred), len(image_id)



## === cell 43
pred[:10]



## === cell 44
assert len(image_id) == len(
    pred
), "Prediction count does not match number of test images."
sub = pd.DataFrame({"image_id": image_id, "label": pred})



## === cell 45
sub.head()



## === cell 46
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
