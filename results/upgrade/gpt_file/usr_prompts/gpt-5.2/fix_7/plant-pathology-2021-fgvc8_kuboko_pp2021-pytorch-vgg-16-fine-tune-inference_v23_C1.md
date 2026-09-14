# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import glob
import os
import os.path as osp

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("high")




## === cell 1
ROOT = "/kaggle/input/plant-pathology-2021-fgvc8/"
TRAIN_CSV = osp.join(ROOT, "train.csv")
SUB_CSV = osp.join(ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = osp.join(ROOT, "train_images")
TEST_IMG_DIR = osp.join(ROOT, "test_images")

assert osp.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert osp.exists(SUB_CSV), f"Missing: {SUB_CSV}"
assert osp.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert osp.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"




## === cell 2
df_train = pd.read_csv(TRAIN_CSV)
df_sub = pd.read_csv(SUB_CSV)




## === cell 3
def to_label(df):
    """
    Function for Label encoding.
    Returns df with labels_n and the fitted LabelEncoder.
    """
    le = LabelEncoder()
    df = df.copy()
    df["labels_n"] = le.fit_transform(df.labels.values)
    return df, le


df_train, le = to_label(df_train)

df_labels_idx = (
    df_train.loc[df_train.duplicated(["labels", "labels_n"]) == False][
        ["labels_n", "labels"]
    ]
    .set_index("labels_n")
    .sort_index()
)

NUM_CLASSES = int(df_train["labels_n"].nunique())
print("num_classes:", NUM_CLASSES)




## === cell 4
try:
    display(df_train.head())
    display(df_train.labels.value_counts().head(10))
    display(df_labels_idx.head(10))
    display(df_sub.head())
except Exception:
    print(df_train.head())
    print(df_train.labels.value_counts().head(10))
    print(df_labels_idx.head(10))
    print(df_sub.head())




## === cell 5
d_set = set()
for k in df_train.labels.unique():
    d_set |= set(k.split(" "))
print(f"num of tokens in label-strings: {len(d_set)}  {d_set}")




## === cell 6
def _get_all_train_paths():
    target_path = osp.join(ROOT, "train_images", "*.jpg")
    return sorted(glob.glob(target_path))


def _get_all_test_paths():
    target_path = osp.join(ROOT, "test_images", "*.jpg")
    return sorted(glob.glob(target_path))


_all_train_paths = _get_all_train_paths()
_train_paths, _val_paths = train_test_split(
    _all_train_paths, test_size=0.25, random_state=0, shuffle=True
)
_test_paths = _get_all_test_paths()

train_list = _train_paths
val_list = _val_paths
test_list = _test_paths

print(f"train data length : {len(train_list)}")
print(f"validation data length : {len(val_list)}")
print(f"test data length : {len(test_list)}")




## === cell 7
class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)




## === cell 8
_image_to_label = dict(
    zip(df_train["image"].values, df_train["labels_n"].astype(np.int64).values)
)


class PlantDataset(data.Dataset):
    def __init__(self, df_train, file_list, transform=None, phase="train"):
        self.df_train = df_train
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self._image_to_label = _image_to_label

        self._precomputed = None
        if self.phase in ["val", "test"]:
            self._precomputed = [None] * len(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def _load_rgb_u8_tensor(self, img_path):
        return read_image(img_path, mode=ImageReadMode.RGB)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        image_name = osp.basename(img_path)

        if self._precomputed is not None:
            cached = self._precomputed[index]
            if cached is None:
                img_u8 = self._load_rgb_u8_tensor(img_path)
                img_transformed = self.transform(img_u8, self.phase)
                self._precomputed[index] = img_transformed
            else:
                img_transformed = cached
        else:
            img_u8 = self._load_rgb_u8_tensor(img_path)
            img_transformed = self.transform(img_u8, self.phase)

        if self.phase in ["train", "val"]:
            label = int(self._image_to_label[image_name])
        else:
            label = -1

        return img_transformed, label, image_name




## === cell 9
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class ImageTransformTensor:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean, std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


train_dataset = PlantDataset(
    df_train, train_list, transform=ImageTransformTensor(size, mean, std), phase="train"
)
val_dataset = PlantDataset(
    df_train, val_list, transform=ImageTransformTensor(size, mean, std), phase="val"
)
test_dataset = PlantDataset(
    df_train, test_list, transform=ImageTransformTensor(size, mean, std), phase="test"
)

index = 0
tr_x, tr_y, tr_n = train_dataset[index]
va_x, va_y, va_n = val_dataset[index]
te_x, te_y, te_n = test_dataset[index]

print("【train dataset】", len(train_dataset), tr_x.shape, tr_y, tr_n)
print("【validation dataset】", len(val_dataset), va_x.shape, va_y, va_n)
print("【test dataset】", len(test_dataset), te_x.shape, te_y, te_n)




## === cell 10
batch_size = 128


def _seed_worker(worker_id):
    base_seed = 0
    s = base_seed + worker_id
    np.random.seed(s)
    torch.manual_seed(s)


_cpu = os.cpu_count() or 2
_num_workers = min(10, max(2, _cpu - 2))
_prefetch = 6 if _num_workers > 0 else None

g = torch.Generator()
g.manual_seed(0)

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)
val_dataloader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)
test_dataloader = data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)

dataloaders_dict = {
    "train": train_dataloader,
    "val": val_dataloader,
    "test": test_dataloader,
}




## === cell 11
weights = models.VGG16_Weights.IMAGENET1K_V1
net = models.vgg16(weights=weights)
net.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)
net.train()

if torch.cuda.is_available():
    net = net.to(memory_format=torch.channels_last)




## === cell 12
criterion = nn.CrossEntropyLoss()




## === cell 13
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = [
    "features.24.weight",
    "features.24.bias",
    "features.26.weight",
    "features.26.bias",
    "features.28.weight",
    "features.28.bias",
]
update_param_names_2 = [
    "classifier.0.weight",
    "classifier.0.bias",
    "classifier.3.weight",
    "classifier.3.bias",
]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_1.append(param)
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
    else:
        param.requires_grad = False

optimizer = optim.SGD(
    [
        {"params": params_to_update_1, "lr": 1e-4},
        {"params": params_to_update_2, "lr": 5e-4},
        {"params": params_to_update_3, "lr": 1e-3},
    ],
    momentum=0.9,
)




## === cell 14
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devices to be used : {device}")
    net.to(device)

    torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")
        print("-------------------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels, _ in tqdm(
                dataloaders_dict[phase], leave=False, miniters=20
            ):
                if device.type == "cuda":
                    inputs = inputs.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device, non_blocking=(device.type == "cuda"))

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return net




## === cell 15
num_epochs = 4
net = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)




## === cell 16
class PlantPredictor:
    def __init__(self, net, label_encoder, dataloaders_dict):
        self.net = net
        self.le = label_encoder
        self.dataloaders_dict = dataloaders_dict
        self.df_submit = pd.DataFrame()

    def __predict_max_labelstr(self, out_cpu_tensor):
        maxid = torch.argmax(out_cpu_tensor, dim=1).numpy()
        label_str = self.le.inverse_transform(maxid)
        return label_str

    def inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Devices to be used : {device}")
        self.net.to(device)
        self.net.eval()

        n_test = len(self.dataloaders_dict["test"].dataset)
        images_all = [None] * n_test
        labels_all = [None] * n_test
        offset = 0

        with torch.no_grad():
            for inputs, _, image_name in tqdm(
                self.dataloaders_dict["test"], leave=False, miniters=20
            ):
                bs = len(image_name)
                if device.type == "cuda":
                    inputs = inputs.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    inputs = inputs.to(device)

                out = self.net(inputs).detach().cpu()
                pred_labels = self.__predict_max_labelstr(out)

                images_all[offset : offset + bs] = list(image_name)
                labels_all[offset : offset + bs] = list(pred_labels)
                offset += bs

        self.df_submit = pd.DataFrame({"image": images_all, "labels": labels_all})




## === cell 17
predictor = PlantPredictor(net, le, dataloaders_dict)
predictor.inference()

df_submit = predictor.df_submit.copy()

df_submit = df_sub[["image"]].merge(df_submit, on="image", how="left")
df_submit["labels"] = df_submit["labels"].fillna("healthy")

print(df_submit.head())
print(df_submit.shape)




## === cell 18
out_path = "/kaggle/working/submission.csv"
df_submit.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(df_submit.columns))
print("Null labels:", int(df_submit["labels"].isna().sum()))
print("Example row:", df_submit.iloc[0].to_dict())
