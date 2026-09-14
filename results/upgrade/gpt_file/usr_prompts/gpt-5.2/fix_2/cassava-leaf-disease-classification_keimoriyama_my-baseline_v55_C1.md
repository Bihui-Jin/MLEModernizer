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

# 5. Code solution

## === cell 0
import os, sys, subprocess, textwrap

wheel_path = "../input/pretrined-models/timm-0.3.3-py3-none-any.whl"
if os.path.exists(wheel_path):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", wheel_path, "-q"],
        check=False,
    )



## === cell 1
import os
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from PIL import Image, ImageDraw

from sklearn import model_selection
from sklearn.model_selection import KFold

import timm



## === cell 2
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)[:10]



## === cell 3
df = pd.read_csv(path + "/train.csv")



## === cell 4
df.head()



## === cell 5
df["path"] = df["image_id"].map(lambda x: path + "/train_images/" + x)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 6
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 7
try:
    ax = train_df.label.value_counts().plot(kind="bar", title="train label counts")
except Exception:
    pass



## === cell 8
try:
    ax = valid_df.label.value_counts().plot(kind="bar", title="valid label counts")
except Exception:
    pass



## === cell 9
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()



## === cell 10
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()



## === cell 11
im = Image.open(train_df["path"][0])
im.size



## === cell 12
try:
    im
except Exception:
    pass



## === cell 13
import matplotlib.image as img  # noqa: F401



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
        path_ = self.df["path"][index]
        label = int(self.df["label"][index])
        with open(path_, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 16
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




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
            ms = min(self.mask_size, max(1, width - 1), max(1, height - 1))
            for _ in range(10):
                start_width.append(random.randrange(0, max(1, width - ms)))
                start_height.append(random.randrange(0, max(1, height - ms)))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + ms, y + ms), fill=(0, 0, 0), outline=(0, 0, 0)
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
pass



## === cell 22
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 23
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)



## === cell 24
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
def calc_correction(model, df_):
    model.eval()
    paths = df_["path"].values
    labels = df_["label"].values.astype(int)
    count = 0
    pred_list = [0, 0, 0, 0, 0]
    with torch.no_grad():
        for image_path, image_label in zip(paths, labels):
            image = Image.open(image_path).convert("RGB")
            image = valid_transform(image).unsqueeze(0).to(device)
            pred = model(image).argmax(1).item()
            pred_list[pred] += 1
            if pred == int(image_label):
                count += 1
    percent = count / len(paths)
    return percent, pred_list




## === cell 27
from matplotlib import pyplot as plt


def plot_losses(epoch_, title, train_losses, valid_losses):
    try:
        y = list(range(len(train_losses)))
        train_loss = plt.plot(y, train_losses)
        valid_loss = plt.plot(y, valid_losses)
        plt.title(title)
        plt.ylabel("loss")
        plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
        plt.show()
    except Exception:
        pass




## === cell 28
def train_model(
    model, dataset_, batch_size_, optimizer, criterion_, scheduler, epoch_, model_title
):
    best_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, valid_index) in enumerate(kf.split(dataset_)):
        print("fold: ", fold)
        train_dataset = Subset(dataset_, train_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size_,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        valid_dataset = Subset(dataset_, valid_index)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size_,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        for ep in range(1, epoch_ + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion_(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data)
            train_loss = train_loss / len(train_loader.sampler)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for data, target in valid_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    output = model(data)
                    pred = output.argmax(1) == target
                    acc.append((pred.float().mean()).item())
                    loss = criterion_(output, target)
                    valid_loss += loss.item() * len(data)

            avg_valid_loss = valid_loss / len(valid_loader.sampler)
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            scheduler.step()

            collection = float(np.mean(acc)) if len(acc) else 0.0
            valid_losses.append(avg_valid_loss)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    ep,
                    train_loss,
                    avg_valid_loss,
                    collection,
                )
            )

    if best_state is not None:
        torch.save(best_state, model_title)
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), model_title)

    return model, train_losses, valid_losses




## === cell 29
def train_models():
    global resNet, ef_model
    model_title = "./res_model.pth"
    resNet, train_losses, valid_losses = train_model(
        resNet,
        dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        model_title,
    )
    print("resNet valid correction:", calc_correction(resNet, valid_df))
    plot_losses(epoch, "resNet losses", train_losses, valid_losses)

    model_title = "./ef_model.pth"
    ef_model, train_losses, valid_losses = train_model(
        ef_model,
        dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        model_title,
    )
    print("ef_model valid correction:", calc_correction(ef_model, valid_df))
    plot_losses(epoch, "ef losses", train_losses, valid_losses)




## === cell 30
train_models()



## === cell 31
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))

ef_model = ef_model.to(device).eval()
resNet = resNet.to(device).eval()




## === cell 32
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




## === cell 33
classifier = CassaveClassifier(resNet, ef_model).to(device).eval()




## === cell 34
def test_rate():
    classifier.eval()
    for rate in range(1, 10):
        paths = valid_df["path"].values
        labels = valid_df["label"].values.astype(int)
        count = 0
        with torch.no_grad():
            for image_path, image_label in zip(paths, labels):
                image = Image.open(image_path).convert("RGB")
                image = valid_transform(image).unsqueeze(0).to(device)
                pred = classifier.test(image, rate / 10).argmax(1).item()
                if pred == int(image_label):
                    count += 1
        percent = count / len(paths)
        print("rate: ", rate / 10, "percent: ", percent)




## === cell 35
pass



## === cell 36
test_images_dir = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 37
sample_sub = pd.read_csv(path + "/sample_submission.csv")
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(test_images_dir, fn) for fn in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test image files. Example: {missing[0]}"
    )



## === cell 38
pred = []
classifier.eval()
with torch.no_grad():
    for pth in image_path:
        image = Image.open(pth).convert("RGB")
        image = valid_transform(image).unsqueeze(0).to(device)
        predict = classifier(image).argmax(1).item()
        pred.append(int(predict))

len(pred), len(image_id)



## === cell 39
pred[:10]



## === cell 40
sub = pd.DataFrame({"image_id": image_id, "label": np.asarray(pred, dtype=np.int64)})



## === cell 41
sub.head()



## === cell 42
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
print(sub.iloc[:3])
