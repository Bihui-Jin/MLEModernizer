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

0.6128739800543971

# 6. Current score

0.16069

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.37593) has done: 'I fix the missing/incorrect model weight paths by removing the dependency on non-existent local `.pth` files and instead using `timm`’s built-in pretrained weights (same architectures, same training loop). I also fix test image discovery to avoid accidentally including the nested `test_images/` directory, which caused `IsADirectoryError` and mismatched `image_id`/`pred` lengths. To keep the core approach intact, the training code is preserved but not required for submission generation; inference use the ensemble classifier already defined. Finally, I ensure the submission matches `sample_submission.csv` order and is written as `submission.csv`.'
- What this solution (achieved 0.16069) has done: 'The timeout is dominated by doing 5-fold cross-validation with 3 epochs for *two* large CNNs (30 epochs worth of full training passes) plus repeated DataLoader construction; that work is far beyond 600 seconds even on GPU. The only way to keep identical inference/evaluation semantics while meeting the limit is to avoid re-training inside the submission run by reliably loading the already-saved weights (and/or failing fast with a clear message if weights are missing), and to reduce overhead in data input/inference (faster DataLoader settings, reuse loaders, channels_last, pinned memory). I also remove/guard all visualization and per-sample preview code that can trigger heavy CPU work and blocking rendering, without affecting model outputs. Core model architectures, losses, transforms, and prediction logic are unchanged.'

# 9. Code solution

## === cell 0
import os, sys

print("Python:", sys.version)
print("timm will be imported from the preinstalled environment.")



## === cell 1
import os
import random
import numpy as np
import pandas as pd
import timm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

Image.MAX_IMAGE_PIXELS = None
try:
    Image.draft = Image.Image.draft
except Exception:
    pass



## === cell 2
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)[:10]



## === cell 3
df = pd.read_csv(path + "/train.csv")
df.head()



## === cell 4
df["path"] = path + "/train_images/" + df["image_id"].astype(str)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()



## === cell 5
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 6
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold

import matplotlib.image as img

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass



## === cell 7
import torchvision.io as tvio
import torchvision.transforms.functional as TF

_SHARED_PIL_CACHE = (
    {}
)  # kept for compatibility with the rest of the notebook; not used for speed.


class CassavaDataset(Dataset):
    def __init__(
        self,
        dataframe,
        transform=None,
        cache_images=True,  # kept for API compatibility; ignored
        cache_size_hw=(512, 512),  # kept for API compatibility; ignored
        shared_cache=None,  # kept for API compatibility; ignored
    ):
        super().__init__()
        self.transform = transform
        self.paths = dataframe["path"].values
        self.labels = dataframe["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index])

        image = tvio.read_image(path, mode=tvio.ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 8
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




## === cell 9
image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        transforms.ConvertImageDtype(
            torch.float32
        ),  # replaces ToTensor() for uint8 tensor input
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=mean, std=std),
    ]
)



## === cell 10
dataset = CassavaDataset(
    train_df,
    train_transform,
    cache_images=False,
    cache_size_hw=(image_size, image_size),
    shared_cache=_SHARED_PIL_CACHE,
)



## === cell 11
import json

label_map_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
with open(label_map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 12
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


unnorm = Unnormalize(mean, std)


def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img.clone())
    plt.imshow(img.permute(1, 2, 0))
    if label is not None:
        plt.title(label_to_name[str(label)])


def display_batch(batch, unnorm=None):
    imgs, labels = batch
    if unnorm:
        imgs = [unnorm(img.clone()) for img in imgs]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))




## === cell 13
if False:
    tensor, label = dataset[0]
    display_img(tensor, unnorm, label)



## === cell 14
import torch
import torch.nn as nn
import torch.nn.functional as F

epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 15
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)

ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)

if device.type == "cuda":
    resNet = resNet.to(memory_format=torch.channels_last)
    ef_model = ef_model.to(memory_format=torch.channels_last)



## === cell 16
fused_ok = device.type == "cuda"
ef_optimizer = torch.optim.AdamW(
    ef_model.parameters(), lr=1e-4, weight_decay=0.0001, fused=fused_ok
)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(
    resNet.parameters(), lr=1e-4, weight_decay=0.0001, fused=fused_ok
)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()



## === cell 17
_NW = min(8, (os.cpu_count() or 2))
_PF = 4 if _NW > 0 else None
_DL_COMMON = dict(
    num_workers=_NW,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_NW > 0),
    prefetch_factor=_PF,
)

_valid_infer_ds = CassavaDataset(
    valid_df,
    transform=valid_transform,
    cache_images=False,
    cache_size_hw=(image_size, image_size),
    shared_cache=_SHARED_PIL_CACHE,
)
_valid_infer_loader = DataLoader(
    _valid_infer_ds,
    batch_size=128,
    shuffle=False,
    **_DL_COMMON,
)


def calc_correction(model, df_):
    model.eval()

    if df_ is valid_df:
        infer_loader = _valid_infer_loader
    else:
        infer_ds = CassavaDataset(
            df_,
            transform=valid_transform,
            cache_images=False,
            cache_size_hw=(image_size, image_size),
            shared_cache=_SHARED_PIL_CACHE,
        )
        infer_loader = DataLoader(
            infer_ds,
            batch_size=128,
            shuffle=False,
            **_DL_COMMON,
        )

    correct = 0
    total = 0
    pred_list = [0, 0, 0, 0, 0]

    with torch.no_grad():
        for images, labels in infer_loader:
            if device.type == "cuda":
                images = images.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
            else:
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images).argmax(1)
            total += labels.numel()
            correct += (preds == labels).sum().item()
            for p in preds.detach().cpu().tolist():
                pred_list[p] += 1

    percent = correct / max(1, total)
    return percent, pred_list




## === cell 18
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
    plt.show()




## === cell 19
import time
import copy


_KF = KFold(n_splits=5, shuffle=False)
_KF_SPLITS = list(_KF.split(np.arange(len(dataset))))


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for fold, (train_index, valid_index) in enumerate(_KF_SPLITS):
        print("fold: ", fold)

        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            **_DL_COMMON,
        )
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size,
            shuffle=False,
            **_DL_COMMON,
        )

        train_den = len(train_loader.sampler)
        valid_den = len(valid_loader.sampler)

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss_sum = 0.0
            valid_loss_sum = 0.0

            model.train()
            for data, target in train_loader:
                if device.type == "cuda":
                    data = data.to(
                        device, non_blocking=True, memory_format=torch.channels_last
                    )
                else:
                    data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += loss.detach().float().item() * len(data)

            train_loss = train_loss_sum / max(1, train_den)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for data, target in valid_loader:
                    if device.type == "cuda":
                        data = data.to(
                            device, non_blocking=True, memory_format=torch.channels_last
                        )
                    else:
                        data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output = model(data)
                        loss = criterion(output, target)

                    pred = output.argmax(1) == target
                    acc.append((pred.float().mean()).item())
                    valid_loss_sum += loss.detach().float().item() * len(data)

            valid_loss = valid_loss_sum / max(1, valid_den)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())

            scheduler.step()

            collection = sum(acc) / max(1, len(acc))
            valid_losses.append(valid_loss)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    ep,
                    train_loss,
                    valid_loss,
                    collection,
                )
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_title)
    return model, train_losses, valid_losses




## === cell 20
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
    print(calc_correction(resNet, valid_df))
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
    print(calc_correction(ef_model, valid_df))
    plot_losses(epoch, "ef losses", train_losses, valid_losses)




## === cell 21
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


classifier = CassaveClassifier(resNet, ef_model).to(device)



## === cell 22
res_path = "./res_model.pth"
ef_path = "./ef_model.pth"

if not (os.path.isfile(res_path) and os.path.isfile(ef_path)):
    raise FileNotFoundError(
        "Required saved weights not found (./res_model.pth and ./ef_model.pth). "
        "Training from scratch is disabled because it cannot finish within the 600s timeout. "
        "Please provide these files in the working directory (e.g., by adding them as a dataset) "
        "to preserve identical model accuracy/semantics."
    )

resNet.load_state_dict(torch.load(res_path, map_location=device), strict=True)
ef_model.load_state_dict(torch.load(ef_path, map_location=device), strict=True)
classifier = CassaveClassifier(resNet, ef_model).to(device)
classifier.eval()

if hasattr(torch, "compile"):
    try:
        classifier = torch.compile(classifier, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(
            "torch.compile unavailable for this model/config; continuing without it:",
            repr(e),
        )



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1729588416.py in <cell line: 0>()
      7 
      8 if not (os.path.isfile(res_path) and os.path.isfile(ef_path)):
----> 9     raise FileNotFoundError(
     10         "Required saved weights not found (./res_model.pth and ./ef_model.pth). "
     11         "Training from scratch is disabled because it cannot finish within the 600s timeout. "

FileNotFoundError: Required saved weights not found (./res_model.pth and ./ef_model.pth). Training from scratch is disabled because it cannot finish within the 600s timeout. Please provide these files in the working directory (e.g., by adding them as a dataset) to preserve identical model accuracy/semantics.

## === cell 23
test_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

test_paths = [
    os.path.join(test_dir, img_id) for img_id in sample_sub["image_id"].tolist()
]

missing = [p for p in test_paths if not os.path.isfile(p)]
print("Missing test files:", len(missing))




## === cell 24
class CassavaTestDataset(Dataset):
    def __init__(
        self,
        paths,
        transform=None,
        cache_images=True,  # kept for API compatibility; ignored
        cache_size_hw=(512, 512),  # kept for API compatibility; ignored
        shared_cache=None,  # kept for API compatibility; ignored
    ):
        super().__init__()
        self.paths = np.asarray(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        image = tvio.read_image(pth, mode=tvio.ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image


test_ds = CassavaTestDataset(
    test_paths,
    transform=valid_transform,
    cache_images=False,
    cache_size_hw=(image_size, image_size),
    shared_cache=_SHARED_PIL_CACHE,
)

test_loader = DataLoader(
    test_ds,
    batch_size=(256 if device.type == "cuda" else 64),
    shuffle=False,
    **_DL_COMMON,
)

pred = []
use_amp = device.type == "cuda"
with torch.no_grad():
    for images in test_loader:
        if device.type == "cuda":
            images = images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
        else:
            images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            predict = classifier(images).argmax(1)
        pred.extend(predict.detach().cpu().to(torch.int64).tolist())



## === cell 25
sub = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": pred})
sub.head()



## === cell 26
sub["label"] = sub["label"].astype(int)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.tail())
