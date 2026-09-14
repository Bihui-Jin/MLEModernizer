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
import random
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import timm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    Image.MAX_IMAGE_PIXELS = None
    ImageFile = __import__("PIL.ImageFile", fromlist=["ImageFile"]).ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

_CPU = os.cpu_count() or 2

DL_NUM_WORKERS = min(8, max(2, _CPU // 2))
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = DL_NUM_WORKERS > 0
PREFETCH_FACTOR = 4 if DL_NUM_WORKERS > 0 else None


@lru_cache(maxsize=4096)
def _load_rgb_base(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def _load_rgb_cached(path: str) -> Image.Image:
    return _load_rgb_base(path).copy()




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
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 7
pass



## === cell 8
pass



## === cell 9
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()



## === cell 10
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass




## === cell 14
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        df_ = dataframe.reset_index(drop=True)
        self.paths = df_["path"].to_numpy()
        self.labels = df_["label"].astype(np.int64).to_numpy()
        self.transform = transform

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index])
        image = _load_rgb_cached(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 15
pass




## === cell 16
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            max_w = max(1, width - self.mask_size)
            max_h = max(1, height - self.mask_size)
            for _ in range(10):
                x = random.randrange(0, max_w)
                y = random.randrange(0, max_h)
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
        transforms.RandomResizedCrop(
            image_size, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        make_mask_image(p=0.5, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)



## === cell 18
train_dataset = CassavaDataset(train_df, train_transform)
valid_dataset = CassavaDataset(valid_df, valid_transform)



## === cell 19
pass



## === cell 20
map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 21
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 22
unnorm = Unnormalize(mean, std)




## === cell 23
def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img)

    plt.imshow(img.permute(1, 2, 0))

    if label is not None:
        plt.title(label_to_name[str(label)])
    plt.axis("off")
    plt.show()




## === cell 24
def display_batch(batch, unnorm=None):
    imgs, labels = batch

    if unnorm is not None:
        imgs = [unnorm(img) for img in imgs]

    ig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))
    plt.show()




## === cell 25
pass



## === cell 26
pass



## === cell 27
pass



## === cell 28
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 29
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)



## === cell 30
ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)



## === cell 31
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 32
def calc_correction(model, df):
    model.eval()
    eval_ds = CassavaDataset(df, transform=valid_transform)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=128,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    count = 0
    pred_hist = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        for data, target in tqdm(eval_loader, total=len(eval_loader)):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            preds = logits.argmax(1)
            count += (preds == target).sum().item()
            pred_hist += torch.bincount(preds.detach().cpu(), minlength=5)

    percent = count / len(eval_ds)
    return percent, pred_hist.tolist()




## === cell 33
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
    plt.show()




## === cell 34
def train_model(
    model,
    train_dataset,
    valid_dataset,
    batch_size,
    optimizer,
    criterion,
    scheduler,
    epoch,
    model_title,
):
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=DL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    for ep in range(1, epoch + 1):
        epoch_start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / (
            len(train_loader.dataset) - (len(train_loader.dataset) % batch_size)
        )
        train_losses.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                preds = output.argmax(1)
                correct += (preds == target).sum().item()
                total += target.numel()
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                model_title,
            )

        scheduler.step()

        collection = correct / total if total else 0.0
        print(
            "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                time.time() - epoch_start_time,
                ep,
                train_loss,
                valid_loss,
                collection,
            )
        )

    if os.path.exists(model_title):
        model.load_state_dict(torch.load(model_title, map_location=device))

    return model, train_losses, valid_losses




## === cell 35
def train_models(resNet, ef_model):
    model_title = "./res_model.pth"
    resNet, train_losses, valid_losses = train_model(
        resNet,
        train_dataset,
        valid_dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        model_title,
    )

    model_title = "./ef_model.pth"
    ef_model, train_losses, valid_losses = train_model(
        ef_model,
        train_dataset,
        valid_dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        model_title,
    )

    return resNet, ef_model




## === cell 36
resNet, ef_model = train_models(resNet, ef_model)



## === cell 37
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))




## === cell 38
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




## === cell 39
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)




## === cell 40
def test_rate():
    for rate in range(1, 10):
        classifier.eval()
        paths = valid_df["path"].tolist()
        labels = valid_df["label"].astype(int).tolist()
        count = 0
        pred_list = [0, 0, 0, 0, 0]
        with torch.no_grad():
            for i in tqdm(range(len(paths))):
                image_path = paths[i]
                image_label = labels[i]
                image = _load_rgb_cached(image_path)
                image = valid_transform(image)
                image = image.unsqueeze(0).to(device)
                pred = int(classifier.test(image, rate / 10).argmax(1).item())
                if 0 <= pred < len(pred_list):
                    pred_list[pred] += 1
                if pred == image_label:
                    count += 1
        percent = count / len(paths)
        print("rate: ", rate / 10)
        print("percent: ", percent)




## === cell 41
pass



## === cell 42
pass



## === cell 43
test_dir = "../input/cassava-leaf-disease-classification/test_images/"
test_dir



## === cell 44
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

available = {
    name: os.path.join(test_dir, name)
    for name in os.listdir(test_dir)
    if name.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(test_dir, name))
}

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in available
]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images referenced in sample_submission, e.g. {missing[:5]}"
    )

image_id = sample_sub["image_id"].tolist()
image_path = [available[iid] for iid in image_id]
len(image_id), len(image_path)




## === cell 45
class TestCassavaDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pth = self.image_paths[idx]
        image = _load_rgb_cached(pth)
        if self.transform is not None:
            image = self.transform(image)
        return image


test_ds = TestCassavaDataset(image_path, transform=valid_transform)

test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False,
    num_workers=DL_NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)

classifier.eval()
pred = []
with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = batch.to(device, non_blocking=True)
        out = classifier(batch)
        pred.extend(out.argmax(1).detach().cpu().tolist())

len(pred)



## === cell 46
pred[:10]



## === cell 47
sub = pd.DataFrame({"image_id": image_id, "label": pred})
sub.head()



## === cell 48
sub.shape, sub.isna().sum()



## === cell 49
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(sub), "rows")
