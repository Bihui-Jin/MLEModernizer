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
import os, sys, glob, warnings

wheel_path = "../input/pretrined-models/timm-0.3.3-py3-none-any.whl"
if os.path.exists(wheel_path):
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", wheel_path]
    )
else:
    warnings.warn(
        f"Optional wheel not found at {wheel_path}; using preinstalled timm.",
        RuntimeWarning,
    )



## === cell 1
import os
import pandas as pd
import timm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
from sklearn import model_selection
import random
import json
import time
import warnings



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
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 6
df.head()



## === cell 7
ex_csv = "../input/extra-images/extra_image.csv"
if os.path.exists(ex_csv):
    ex_df = pd.read_csv(ex_csv, index_col=0)
    ex_df["path"] = ex_df["image_id"].map(
        lambda x: "../input/extra-images/images/images/" + str(x)
    )
    ex_df = ex_df.drop(columns=["image_id"])
else:
    ex_df = pd.DataFrame(columns=["label", "path"])
ex_df.head()



## === cell 8
ex_df



## === cell 9
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)

if len(ex_df) > 0:
    train_df = pd.concat([train_df, ex_df], ignore_index=True)



## === cell 10
train_df.label.value_counts().plot(kind="bar")



## === cell 11
valid_df.label.value_counts().plot(kind="bar")



## === cell 12
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()



## === cell 13
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()



## === cell 14
im = Image.open(train_df["path"][0])
im



## === cell 15
im



## === cell 16
pass




## === cell 17
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df["path"])

    def __getitem__(self, index):
        path = self.df.loc[index, "path"]
        label = int(self.df.loc[index, "label"])
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 18
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            if width <= self.mask_size or height <= self.mask_size:
                return image
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




## === cell 19
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 20
resNet = timm.create_model("resnet50", pretrained=False)
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)

res_cfg = timm.data.resolve_data_config({}, model=resNet)
ef_cfg = timm.data.resolve_data_config({}, model=ef_model)

image_size = int(res_cfg.get("input_size", (3, 224, 224))[1])

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.5, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=res_cfg["mean"], std=res_cfg["std"]),
    ]
)

res_valid_transform = timm.data.create_transform(**res_cfg, is_training=False)
ef_valid_transform = timm.data.create_transform(**ef_cfg, is_training=False)

res_inference_transform = timm.data.create_transform(**res_cfg, is_training=False)
ef_inference_transform = timm.data.create_transform(**ef_cfg, is_training=False)

inference_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=res_cfg["mean"], std=res_cfg["std"]),
    ]
)



## === cell 21
dataset = CassavaDataset(train_df, train_transform)



## === cell 22
pass



## === cell 23
map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 24
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 25
unnorm = Unnormalize(res_cfg["mean"], res_cfg["std"])




## === cell 26
def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img)
    plt.imshow(img.permute(1, 2, 0))
    if label is not None:
        plt.title(label_to_name[str(label)])




## === cell 27
def display_batch(batch, unnorm=None):
    imgs, labels = batch
    if unnorm:
        unnorm_imgs = []
        for img in imgs:
            unnorm_imgs.append(unnorm(img))
        imgs = unnorm_imgs
    ig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))




## === cell 28
tensor, label = dataset[0]
display_img(tensor, unnorm, label)



## === cell 29
loader = DataLoader(dataset, 16, shuffle=True, num_workers=2)
display_batch(next(iter(loader)))



## === cell 30
epoch = 3
batch_size = 16



## === cell 31
res_w = "../input/pretrined-models/models/models/pretrained_resNet.pth"
if os.path.exists(res_w):
    resNet.load_state_dict(torch.load(res_w, map_location="cpu"))
else:
    try:
        resNet = timm.create_model("resnet50", pretrained=True)
        res_cfg = timm.data.resolve_data_config({}, model=resNet)
        res_valid_transform = timm.data.create_transform(**res_cfg, is_training=False)
        res_inference_transform = timm.data.create_transform(
            **res_cfg, is_training=False
        )
    except Exception as e:
        warnings.warn(
            f"Could not load timm pretrained resnet50; continuing with random init. Error: {e}",
            RuntimeWarning,
        )

resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)



## === cell 32
ef_w = "../input/pretrined-models/models/models/pretrained_ef_model.pth"
if os.path.exists(ef_w):
    ef_model.load_state_dict(torch.load(ef_w, map_location="cpu"))
else:
    try:
        ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
        ef_cfg = timm.data.resolve_data_config({}, model=ef_model)
        ef_valid_transform = timm.data.create_transform(**ef_cfg, is_training=False)
        ef_inference_transform = timm.data.create_transform(**ef_cfg, is_training=False)
    except Exception as e:
        warnings.warn(
            f"Could not load timm pretrained tf_efficientnet_b2_ns; continuing with random init. Error: {e}",
            RuntimeWarning,
        )

ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)



## === cell 33
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 34
def calc_correction(model, df_, valid_transform_):
    model.eval()
    paths = df_["path"].reset_index(drop=True)
    labels = df_["label"].reset_index(drop=True)
    count = 0
    pred_list = [0, 0, 0, 0, 0]
    for i in tqdm(range(len(paths))):
        image_path = paths[i]
        image_label = int(labels[i])
        image = Image.open(image_path).convert("RGB")
        image = valid_transform_(image)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(image).argmax(1).item()
        pred_list[pred] += 1
        if pred == image_label:
            count += 1
    percent = count / len(paths)
    return percent, pred_list




## === cell 35
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
    plt.show()




## === cell 36
def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        print("fold: ", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(
            train_dataset, batch_size, shuffle=True, num_workers=4
        )
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(
            valid_dataset, batch_size, shuffle=False, num_workers=2
        )

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
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
                    acc.append((pred.float().mean()).item())
                    loss = criterion(output, target)
                    valid_loss += loss.item() * len(data)

            valid_loss = valid_loss / len(valid_loader.sampler)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            scheduler.step()
            valid_losses.append(valid_loss)

            collection = sum(acc) / max(len(acc), 1)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    ep,
                    train_loss,
                    valid_loss,
                    collection,
                )
            )

    if "best_state" in locals():
        torch.save(best_state, model_title)
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), model_title)

    return model, train_losses, valid_losses




## === cell 37
def train_models(resNet, ef_model):
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
    print(calc_correction(resNet, valid_df, res_valid_transform))
    title = "resNet losses"
    plot_losses(epoch, title, train_losses, valid_losses)

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
    print(calc_correction(ef_model, valid_df, ef_valid_transform))
    title = "ef losses"
    plot_losses(epoch, title, train_losses, valid_losses)




## === cell 38
pass



## === cell 39
ef_local = "./ef_model.pth"
res_local = "./res_model.pth"

ef_input = "../input/pretrined-models/models/models/ef_model.pth"
res_input = "../input/pretrined-models/models/models/res_model.pth"

if os.path.exists(ef_input):
    ef_model.load_state_dict(torch.load(ef_input, map_location=device))
elif os.path.exists(ef_local):
    ef_model.load_state_dict(torch.load(ef_local, map_location=device))

if os.path.exists(res_input):
    resNet.load_state_dict(torch.load(res_input, map_location=device))
elif os.path.exists(res_local):
    resNet.load_state_dict(torch.load(res_local, map_location=device))




## === cell 40
def fine_tune_full_data(
    model, optimizer, scheduler, criterion, full_df, transform, epoch, batch_size
):
    full_ds = CassavaDataset(full_df, transform)
    full_loader = DataLoader(
        full_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    model.train()
    for ep in range(1, epoch + 1):
        epoch_loss = 0.0
        for data, target in full_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(data)
        scheduler.step()
        print(
            f"Full-data fine-tune epoch {ep}/{epoch} - loss: {epoch_loss/len(full_loader.dataset):.4f}"
        )
    return model


need_train_res = not (os.path.exists(res_input) or os.path.exists(res_local))
need_train_ef = not (os.path.exists(ef_input) or os.path.exists(ef_local))

if need_train_res or need_train_ef:
    train_models(resNet, ef_model)
    resNet = fine_tune_full_data(
        resNet,
        resNet_optimizer,
        resNet_scheduler,
        criterion,
        train_df,
        train_transform,
        epoch=epoch,
        batch_size=batch_size,
    )
    ef_model = fine_tune_full_data(
        ef_model,
        ef_optimizer,
        ef_scheduler,
        criterion,
        train_df,
        train_transform,
        epoch=epoch,
        batch_size=batch_size,
    )
    torch.save(resNet.state_dict(), res_local)
    torch.save(ef_model.state_dict(), ef_local)




## === cell 41
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.5 * x1 + 0.5 * x2

    def test(self, x, rate):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        p = rate * x1 + (1 - rate) * x2
        return p




## === cell 42
classifier = CassaveClassifier(resNet, ef_model).to(device)




## === cell 43
def test_rate():
    for rate in range(1, 10):
        classifier.eval()
        paths = valid_df["path"].reset_index(drop=True)
        labels = valid_df["label"].reset_index(drop=True)
        count = 0
        pred_list = [0, 0, 0, 0, 0]
        for i in tqdm(range(len(paths))):
            image_path = paths[i]
            image_label = int(labels[i])
            image = Image.open(image_path).convert("RGB")

            image = res_valid_transform(image)

            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = classifier.test(image, rate / 10).argmax(1).item()
            pred_list[pred] += 1
            if pred == image_label:
                count += 1
        percent = count / len(paths)
        print("rate: ", rate / 10)
        print("percent: ", percent)




## === cell 44
pass



## === cell 45
pass



## === cell 46
test_dir = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 47
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(test_dir, img_id) for img_id in image_id]

nested_dir = os.path.join(test_dir, "test_images")
if not os.path.exists(image_path[0]) and os.path.isdir(nested_dir):
    image_path = [os.path.join(nested_dir, img_id) for img_id in image_id]

missing = [p for p in image_path if not os.path.exists(p)]
if len(missing) > 0:
    warnings.warn(
        f"{len(missing)} test image paths do not exist (showing up to 3): {missing[:3]}",
        RuntimeWarning,
    )

len(image_id), len(image_path)



## === cell 48
classifier.eval()
resNet.eval()
ef_model.eval()


class TestImageDataset(Dataset):
    def __init__(self, paths, transform, fallback_size):
        self.paths = list(paths)
        self.transform = transform
        self.fallback_size = int(fallback_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        if not os.path.isfile(pth):
            x = torch.zeros(
                3, self.fallback_size, self.fallback_size, dtype=torch.float32
            )
            ok = 0
            return x, ok
        img = Image.open(pth).convert("RGB")
        x = self.transform(img)
        ok = 1
        return x, ok


res_sz = int(res_cfg.get("input_size", (3, 224, 224))[1])
ef_sz = int(ef_cfg.get("input_size", (3, 260, 260))[1])

test_ds_res = TestImageDataset(
    image_path, res_inference_transform, fallback_size=res_sz
)
test_loader_res = DataLoader(
    test_ds_res,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_ds_ef = TestImageDataset(image_path, ef_inference_transform, fallback_size=ef_sz)
test_loader_ef = DataLoader(
    test_ds_ef,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

pred = []
with torch.no_grad():
    for (xb_res, okb_res), (xb_ef, okb_ef) in tqdm(
        zip(test_loader_res, test_loader_ef),
        total=len(test_loader_res),
    ):
        okb = (okb_res & okb_ef).to(torch.int64)

        xb_res = xb_res.to(device, non_blocking=True)
        xb_ef = xb_ef.to(device, non_blocking=True)

        logits_res = resNet(xb_res)
        logits_ef = ef_model(xb_ef)

        prob = 0.5 * logits_res.softmax(dim=1) + 0.5 * logits_ef.softmax(dim=1)
        batch_pred = prob.argmax(1).detach().cpu().tolist()

        okb = okb.detach().cpu().tolist()
        for pr, ok in zip(batch_pred, okb):
            pred.append(int(pr) if ok == 1 else 0)

len(pred), pred[:5]



## === cell 49
pred[:20]



## === cell 50
sub = pd.DataFrame({"image_id": image_id, "label": pred})
sub.head()



## === cell 51
sub.shape, sub.isna().sum()



## === cell 52
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
