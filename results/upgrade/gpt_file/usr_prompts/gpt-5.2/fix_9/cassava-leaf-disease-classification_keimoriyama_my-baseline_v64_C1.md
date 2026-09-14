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
os.listdir(path)[:10]



## === cell 2
df = pd.read_csv(path + "/train.csv")



## === cell 3
df.head()



## === cell 4
df["path"] = path + "/train_images/" + df["image_id"].astype(str)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 5
df.head()



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
    def __init__(self, dataframe, transform=None, cache_images=True):
        super().__init__()
        df = dataframe.reset_index(drop=True)
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
        self.cache_images = cache_images
        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self.paths)

    def _load_pil_rgb(self, path):
        if self.cache_images:
            im = self._cache.get(path, None)
            if im is not None:
                return im
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.cache_images:
            self._cache[path] = image
        return image

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = self._load_pil_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




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
                start_width.append(random.randrange(0, max(1, width - self.mask_size)))
                start_height.append(
                    random.randrange(0, max(1, height - self.mask_size))
                )
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 18
image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)



## === cell 19
train_dataset_full = CassavaDataset(train_df, train_transform, cache_images=True)
valid_dataset_full = CassavaDataset(valid_df, valid_transform, cache_images=True)



## === cell 20
pass



## === cell 21
import json

map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 22
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 23
unnorm = Unnormalize(mean, std)




## === cell 24
def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img)

    plt.imshow(img.permute(1, 2, 0))

    if label is not None:
        plt.title(label_to_name[str(label)])




## === cell 25
def display_batch(batch, unnorm=None):
    imgs, labels = batch

    if unnorm:
        imgs = [unnorm(img) for img in imgs]

    ig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))




## === cell 26
tensor, label = train_dataset_full[0]
display_img(tensor, unnorm, label)



## === cell 27
DO_VIS = False
if DO_VIS:
    loader = DataLoader(train_dataset_full, 16, shuffle=True, num_workers=2)
    display_batch(next(iter(loader)), unnorm=unnorm)



## === cell 28
import torch
import torch.nn as nn
import torch.nn.functional as F



## === cell 29
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _default_num_workers():
    if not torch.cuda.is_available():
        return 0
    return max(2, min(4, (os.cpu_count() or 4) // 2))


DL_NUM_WORKERS = _default_num_workers()
DL_PREFETCH = 4 if DL_NUM_WORKERS > 0 else None
print("DataLoader num_workers:", DL_NUM_WORKERS, "prefetch_factor:", DL_PREFETCH)



## === cell 30
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)



## === cell 31
ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)



## === cell 32
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 33
def calc_correction(model, df=None, dataset=None, batch_size_eval=64, num_workers=None):
    model.eval()
    if num_workers is None:
        num_workers = DL_NUM_WORKERS

    if dataset is None:
        ds = CassavaDataset(df, transform=valid_transform, cache_images=True)
    else:
        ds = dataset

    loader = DataLoader(
        ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(DL_PREFETCH if num_workers > 0 else None),
    )

    correct = 0
    pred_list = [0, 0, 0, 0, 0]

    with torch.no_grad():
        for data, target in tqdm(loader, total=len(loader)):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(data).argmax(1)
            correct += (pred == target).sum().item()
            binc = torch.bincount(pred.detach().cpu(), minlength=5).tolist()
            for i in range(5):
                pred_list[i] += int(binc[i])

    percent = correct / len(ds)
    return percent, pred_list




## === cell 34
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
    plt.show()




## === cell 35
import time


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_model_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    use_channels_last = torch.cuda.is_available()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    for fold, (train_index, valid_index) in enumerate(kf.split(range(len(dataset)))):
        print("fold: ", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=DL_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(DL_NUM_WORKERS > 0),
            prefetch_factor=(DL_PREFETCH if DL_NUM_WORKERS > 0 else None),
        )
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size,
            shuffle=False,
            num_workers=DL_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(DL_NUM_WORKERS > 0),
            prefetch_factor=(DL_PREFETCH if DL_NUM_WORKERS > 0 else None),
        )

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                if use_channels_last:
                    data = data.contiguous(memory_format=torch.channels_last)
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for data, target in valid_loader:
                    if use_channels_last:
                        data = data.contiguous(memory_format=torch.channels_last)
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    output = model(data)
                    pred = output.argmax(1) == target
                    acc.append((pred.float().mean()).item())

                    loss = criterion(output, target)
                    valid_loss += loss.item() * len(data)

            valid_loss_mean = valid_loss / len(valid_loader.dataset)
            if valid_loss_mean < best_loss:
                best_loss = valid_loss_mean
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            scheduler.step()

            collection = float(sum(acc) / max(1, len(acc)))
            valid_losses.append(valid_loss_mean)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    ep,
                    train_loss,
                    valid_loss_mean,
                    collection,
                )
            )

    if best_model_state is not None:
        torch.save(best_model_state, model_title)
        model.load_state_dict(best_model_state, strict=True)

    return model, train_losses, valid_losses




## === cell 36
def train_models(resNet, ef_model):
    model_title = "./res_model.pth"
    resNet, train_losses, valid_losses = train_model(
        resNet,
        train_dataset_full,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        model_title,
    )
    print(calc_correction(resNet, dataset=valid_dataset_full))

    DO_PLOT = False
    if DO_PLOT:
        title = "resNet losses"
        plot_losses(epoch, title, train_losses, valid_losses)

    model_title = "./ef_model.pth"
    ef_model, train_losses, valid_losses = train_model(
        ef_model,
        train_dataset_full,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        model_title,
    )
    print(calc_correction(ef_model, dataset=valid_dataset_full))
    if DO_PLOT:
        title = "ef losses"
        plot_losses(epoch, title, train_losses, valid_losses)




## === cell 37
DO_TRAIN = True
if DO_TRAIN:
    train_models(resNet, ef_model)



## === cell 38
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(
        torch.load("./ef_model.pth", map_location=device), strict=True
    )
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(
        torch.load("./res_model.pth", map_location=device), strict=True
    )




## === cell 39
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




## === cell 40
classifier = CassaveClassifier(resNet, ef_model).to(device)




## === cell 41
def _batched_logits_two_models(
    model_a, model_b, df=None, dataset=None, batch_size_eval=64, num_workers=None
):
    if num_workers is None:
        num_workers = DL_NUM_WORKERS

    if dataset is None:
        ds = CassavaDataset(df, transform=valid_transform, cache_images=True)
    else:
        ds = dataset

    loader = DataLoader(
        ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(DL_PREFETCH if num_workers > 0 else None),
    )

    model_a.eval()
    model_b.eval()

    logits_a_chunks = []
    logits_b_chunks = []
    labels_chunks = []

    use_channels_last = torch.cuda.is_available()

    with torch.no_grad():
        for data, target in loader:
            if use_channels_last:
                data = data.contiguous(memory_format=torch.channels_last)
            data = data.to(device, non_blocking=True)
            la = model_a(data).detach().cpu()
            lb = model_b(data).detach().cpu()
            logits_a_chunks.append(la)
            logits_b_chunks.append(lb)
            labels_chunks.append(target.clone())

    logits_a = torch.cat(logits_a_chunks, dim=0)
    logits_b = torch.cat(logits_b_chunks, dim=0)
    labels = torch.cat(labels_chunks, dim=0).to(torch.long)
    return logits_a, logits_b, labels


def find_best_rate_on_valid(classifier, valid_df=None, valid_dataset=None):
    logits_res, logits_ef, labels = _batched_logits_two_models(
        classifier.model,
        classifier.ef_model,
        df=valid_df,
        dataset=valid_dataset,
        batch_size_eval=64,
        num_workers=DL_NUM_WORKERS,
    )

    best_rate = 0.5
    best_acc = -1.0

    for rate_i in range(1, 10):
        rate = rate_i / 10.0
        p = rate * logits_res + (1.0 - rate) * logits_ef
        pred = p.argmax(1)
        acc = (pred == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_rate = rate

    print("Best ensemble rate on valid_df:", best_rate, "valid acc:", best_acc)
    return best_rate, best_acc




## === cell 42
pass



## === cell 43
path = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 44
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(path, fname) for fname in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
print("Expected test images:", len(image_id))
print("Missing files:", len(missing))
if len(missing) > 0:
    print("First missing example:", missing[0])



## === cell 45
best_rate, _ = find_best_rate_on_valid(classifier, valid_dataset=valid_dataset_full)




## === cell 46
class CassavaTestDataset(Dataset):
    def __init__(self, image_paths, transform=None, cache_images=True):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.cache_images = cache_images
        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self.image_paths)

    def _load_pil_rgb(self, path):
        if self.cache_images:
            im = self._cache.get(path, None)
            if im is not None:
                return im
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.cache_images:
            self._cache[path] = image
        return image

    def __getitem__(self, idx):
        pth = self.image_paths[idx]
        image = self._load_pil_rgb(pth)
        if self.transform is not None:
            image = self.transform(image)
        return image


classifier.eval()
test_ds = CassavaTestDataset(image_path, transform=valid_transform, cache_images=True)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=DL_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_NUM_WORKERS > 0),
    prefetch_factor=(DL_PREFETCH if DL_NUM_WORKERS > 0 else None),
)

use_channels_last = torch.cuda.is_available()
pred = []
with torch.no_grad():
    for data in tqdm(test_loader, total=len(test_loader)):
        if use_channels_last:
            data = data.contiguous(memory_format=torch.channels_last)
        data = data.to(device, non_blocking=True)
        out = classifier.test(data, best_rate)
        pred.extend(out.argmax(1).detach().cpu().to(torch.int64).tolist())



## === cell 47
pred[:10], len(pred)



## === cell 48
sub = pd.DataFrame({"image_id": image_id, "label": pred})



## === cell 49
sub.head(), sub.shape



## === cell 50
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Submission columns:", list(sub.columns))
