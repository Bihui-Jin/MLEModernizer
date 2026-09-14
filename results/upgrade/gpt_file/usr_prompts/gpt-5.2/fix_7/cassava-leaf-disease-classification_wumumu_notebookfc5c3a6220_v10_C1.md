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
import json
import os
import tqdm
import time
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models

map_json_path = (
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
train_data_path = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_data_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_pretrained = False
batch_size = 8
img_resize = (100, 100)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

try:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

CHANNELS_LAST = torch.cuda.is_available()

try:
    import torchvision.io as tvio

    _HAS_TVIO = True
except Exception:
    tvio = None
    _HAS_TVIO = False

import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode


class CUDAPrefetcher:
    def __init__(self, loader, device, channels_last=False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last and (device == "cuda")
        self.stream = torch.cuda.Stream() if device == "cuda" else None

    def __iter__(self):
        if self.device != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        with torch.cuda.stream(self.stream):
            next_batch = next(it, None)
            if next_batch is None:
                return
            images, labels = next_batch
            images = images.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)

        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = (images, labels)

            with torch.cuda.stream(self.stream):
                next_batch = next(it, None)
                if next_batch is None:
                    next_images = next_labels = None
                else:
                    next_images, next_labels = next_batch
                    next_images = next_images.to(self.device, non_blocking=True)
                    if self.channels_last:
                        next_images = next_images.contiguous(
                            memory_format=torch.channels_last
                        )
                    next_labels = next_labels.to(self.device, non_blocking=True)

            yield batch
            if next_batch is None:
                break
            images, labels = next_images, next_labels


class CUDAPrefetcherTest:
    def __init__(self, loader, device, channels_last=False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last and (device == "cuda")
        self.stream = torch.cuda.Stream() if device == "cuda" else None

    def __iter__(self):
        if self.device != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        with torch.cuda.stream(self.stream):
            next_batch = next(it, None)
            if next_batch is None:
                return
            images, names = next_batch
            images = images.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = (images, names)

            with torch.cuda.stream(self.stream):
                next_batch = next(it, None)
                if next_batch is None:
                    next_images = next_names = None
                else:
                    next_images, next_names = next_batch
                    next_images = next_images.to(self.device, non_blocking=True)
                    if self.channels_last:
                        next_images = next_images.contiguous(
                            memory_format=torch.channels_last
                        )

            yield batch
            if next_batch is None:
                break
            images, names = next_images, next_names




## === cell 1
class CassavaDataset(Dataset):
    def __init__(self, data_path, image_list, label_dict, transforms=None):
        self.data_path = data_path
        self.image_list = image_list
        self.label_dict = label_dict
        self.transforms = transforms

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = int(self.label_dict[image_name])
        fp = os.path.join(self.data_path, image_name)

        if _HAS_TVIO:
            img = tvio.read_image(fp)  # uint8, CxHxW
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] == 4:
                img = img[:3]
            image = img  # tensor
        else:
            with Image.open(fp) as img:
                image = img.convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_list)


class get_test_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.image_list = sorted(os.listdir(data_path))

    def __getitem__(self, index):
        image_name = self.image_list[index]
        fp = os.path.join(self.data_path, image_name)

        if _HAS_TVIO:
            img = tvio.read_image(fp)
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] == 4:
                img = img[:3]
            image = img
        else:
            with Image.open(fp) as img:
                image = img.convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        return image, image_name

    def __len__(self):
        return len(self.image_list)


if _HAS_TVIO:
    train_transforms = T.Compose(
        [
            T.Resize(
                img_resize, interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.ToDtype(torch.float32, scale=True),  # uint8 -> float in [0,1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.RandomErasing(),
        ]
    )
    eval_transforms = T.Compose(
        [
            T.Resize(
                img_resize, interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
else:
    train_transforms = transforms.Compose(
        [
            transforms.Resize(img_resize),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

label_csv = pd.read_csv(train_csv_path)
label_dict = label_csv.set_index("image_id")["label"].to_dict()

images_name_list = label_csv["image_id"].tolist()
images_name_list = [
    x for x in images_name_list if os.path.exists(os.path.join(train_data_path, x))
]
stratify_labels = label_csv.set_index("image_id").loc[images_name_list]["label"].values

train_image, val_image = train_test_split(
    images_name_list,
    train_size=0.9,
    random_state=0,
    stratify=stratify_labels,
)

train_dataset = CassavaDataset(
    train_data_path, train_image, label_dict, train_transforms
)
validation_dataset = CassavaDataset(
    train_data_path, val_image, label_dict, eval_transforms
)

cpu_cnt = os.cpu_count() or 2
if device == "cuda":
    num_workers = min(8, max(2, cpu_cnt))
    prefetch_factor = 8
else:
    num_workers = min(4, max(2, cpu_cnt))
    prefetch_factor = 4

pin = torch.cuda.is_available()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
)

test_dataset = get_test_dataset(test_data_path, eval_transforms)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
)

train_loader_prefetch = CUDAPrefetcher(
    train_dataloader, device, channels_last=CHANNELS_LAST
)
val_loader_prefetch = CUDAPrefetcher(
    validation_dataloader, device, channels_last=CHANNELS_LAST
)
test_loader_prefetch = CUDAPrefetcherTest(
    test_dataloader, device, channels_last=CHANNELS_LAST
)



## === cell 2
with open(map_json_path, "r") as f:
    map_json_data = json.load(f)

map_json_data



## === cell 3
if False:
    plt.figure(figsize=(20, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        image = train_dataset[i][0]
        image = image * 0.5 + 0.5
        transforms_to_pil = transforms.ToPILImage()
        if torch.is_tensor(image):
            image = transforms_to_pil(image)
        plt.title(map_json_data[str(train_dataset[i][1])])
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])



## === cell 4
model = models.vgg16_bn(pretrained=model_pretrained)
sequential = list(model.classifier[:3])
sequential.append(nn.Linear(4096, 5))
model.classifier = nn.Sequential(*sequential)

if CHANNELS_LAST:
    model = model.to(memory_format=torch.channels_last)

model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5
)
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass




## === cell 5
def _update_class_counts_gpu(
    labels, preds, class_num, total_per_class, correct_per_class
):
    total_per_class += torch.bincount(labels, minlength=class_num)
    correct_mask = preds == labels
    if correct_mask.any():
        correct_per_class += torch.bincount(labels[correct_mask], minlength=class_num)


def train(cur_epoch, dataloader, compute_grid=True):
    if compute_grid:
        model.train()
    else:
        model.eval()

    tq_description = "epoch %d" % cur_epoch
    tqbar = tqdm.tqdm(
        dataloader,
        total=(
            len(train_dataloader)
            if dataloader is train_loader_prefetch
            else (
                len(validation_dataloader)
                if dataloader is val_loader_prefetch
                else len(dataloader)
            )
        ),
        mininterval=0.5,
        leave=False,
        desc=tq_description,
    )

    total_loss = 0.0
    correct = 0
    total = 0

    class_num = 5
    if device == "cuda":
        total_per_class = torch.zeros(class_num, dtype=torch.long, device=device)
        correct_per_class = torch.zeros(class_num, dtype=torch.long, device=device)
    else:
        total_per_class = torch.zeros(class_num, dtype=torch.long, device="cpu")
        correct_per_class = torch.zeros(class_num, dtype=torch.long, device="cpu")

    if compute_grid:
        context = torch.enable_grad()
    else:
        context = torch.inference_mode()

    with context:
        for images, labels in tqbar:
            if images.device.type != device:
                images = images.to(device, non_blocking=True)
                if CHANNELS_LAST and device == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
            if labels.device.type != device:
                labels = labels.to(device, non_blocking=True)

            model_out = model(images)
            loss = criterion(model_out, labels)
            preds = torch.argmax(model_out, dim=1)

            if compute_grid:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = labels.shape[0]
            total += bs
            if device == "cuda":
                correct += int((preds == labels).sum().item())
            else:
                correct += int((preds == labels).sum().detach().cpu())

            total_loss += float(loss.detach().item())

            if device == "cuda":
                _update_class_counts_gpu(
                    labels, preds, class_num, total_per_class, correct_per_class
                )
            else:
                labels_cpu = labels.detach().cpu()
                preds_cpu = preds.detach().cpu()
                total_per_class += torch.bincount(labels_cpu, minlength=class_num)
                correct_mask = preds_cpu == labels_cpu
                if correct_mask.any():
                    correct_per_class += torch.bincount(
                        labels_cpu[correct_mask], minlength=class_num
                    )

    preds_arr = np.empty((0,), dtype=np.int64)
    labels_arr = np.empty((0,), dtype=np.int64)

    if device == "cuda":
        total_per_class_cpu = total_per_class.detach().cpu().numpy()
        correct_per_class_cpu = correct_per_class.detach().cpu().numpy()
    else:
        total_per_class_cpu = total_per_class.numpy()
        correct_per_class_cpu = correct_per_class.numpy()

    stats = {
        "total": total,
        "correct": correct,
        "total_per_class": total_per_class_cpu,
        "correct_per_class": correct_per_class_cpu,
    }
    return preds_arr, labels_arr, total_loss, stats


def generate_submission_csv():
    tq_description = "generate csv"
    tqbar = tqdm.tqdm(
        test_loader_prefetch,
        total=len(test_dataloader),
        mininterval=0.5,
        leave=False,
        desc=tq_description,
    )

    model.load_state_dict(torch.load("model.pkl", map_location=device))
    model.eval()

    names_list = []
    preds_chunks = []

    with torch.inference_mode():
        for images, names in tqbar:
            if images.device.type != device:
                images = images.to(device, non_blocking=True)
                if CHANNELS_LAST and device == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)

            model_out = model(images)
            preds = torch.argmax(model_out, dim=1)

            names_list.extend(list(names))
            preds_chunks.append(preds.detach().cpu())

    preds_list = torch.cat(preds_chunks).tolist() if preds_chunks else []

    submission = pd.DataFrame({"image_id": names_list, "label": preds_list})
    sample_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
        submission["label"] = submission["label"].fillna(0).astype(int)

    submission.to_csv("submission.csv", index=False)




## === cell 6
def compute_recall_from_counts(total_per_class, correct_per_class):
    total_per_class = total_per_class.astype(np.float64, copy=False)
    correct_per_class = correct_per_class.astype(np.float64, copy=False)
    recall = np.divide(
        correct_per_class,
        total_per_class,
        out=np.zeros_like(correct_per_class, dtype=np.float64),
        where=(total_per_class != 0),
    )
    return recall


def compute_accuracy_from_counts(total, correct):
    return float(correct / total) if total else 0.0


def do_train(epoch):
    train_loss_list = []
    train_accuracy_list = []
    train_recall_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_recall_list = []

    best_accuracy = [-1, -1]  # (epoch, value)
    train_image_num = len(train_dataset)
    val_image_num = len(validation_dataset)

    print("info:")
    print("train image number: ", train_image_num)
    print("validation image number:", val_image_num)
    print("train on: %s" % device)
    print("train epoch: %d" % epoch)

    for i in range(epoch):
        _, _, total_loss, stats = train(i, train_loader_prefetch, True)
        accuracy = compute_accuracy_from_counts(stats["total"], stats["correct"])
        recall = compute_recall_from_counts(
            stats["total_per_class"], stats["correct_per_class"]
        )
        train_loss_list.append(total_loss)
        train_accuracy_list.append(accuracy)
        train_recall_list.append(recall)
        print("train loss: %f" % total_loss)
        print("train accuracy: %f" % accuracy)
        print("train recall:", recall)

        _, _, total_loss, stats = train(i, val_loader_prefetch, False)
        accuracy = compute_accuracy_from_counts(stats["total"], stats["correct"])
        recall = compute_recall_from_counts(
            stats["total_per_class"], stats["correct_per_class"]
        )
        val_loss_list.append(total_loss)
        val_accuracy_list.append(accuracy)
        val_recall_list.append(recall)
        print("val loss: %f" % total_loss)
        print("val accuracy: %f" % accuracy)
        print("val recall:", recall)

        if best_accuracy[1] < accuracy:
            best_accuracy[0] = i
            best_accuracy[1] = accuracy
            torch.save(model.state_dict(), "model.pkl")

    if False:
        plt.figure()
        plt.plot(train_loss_list)
        plt.plot(val_loss_list)
        plt.title("loss")
        plt.legend(labels=["train", "validation"])

        plt.figure()
        plt.plot(train_accuracy_list)
        plt.plot(val_accuracy_list)
        plt.title("accuracy")
        plt.legend(labels=["train", "validation"])

        plt.figure()
        plt.plot([r.mean() for r in train_recall_list])
        plt.plot([r.mean() for r in val_recall_list])
        plt.title("mean recall")
        plt.legend(labels=["train", "validation"])


do_train(20)
generate_submission_csv()
print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").head())
