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

try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

try:
    import torch.multiprocessing as mp

    mp.set_start_method("fork", force=True)
except Exception:
    pass

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except Exception:
    pass

USE_CUDA = torch.cuda.is_available()
CPU_COUNT = os.cpu_count() or 4


def _default_num_workers(kind: str):
    if kind == "train":
        return min(8, CPU_COUNT)
    if kind == "valid":
        return min(4, CPU_COUNT)
    if kind == "test":
        return min(8, CPU_COUNT)
    return min(4, CPU_COUNT)


def _prefetch_factor_for(nw: int):
    return 4 if nw > 0 else None


def load_state_dict_flexible(model: nn.Module, checkpoint, strict: bool = True):
    """
    Correctness-preserving: accepts common checkpoint formats:
    - raw state_dict
    - {"state_dict": ...}
    - DataParallel-prefixed keys ("module.")
    """
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location="cpu")
    sd = checkpoint
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and isinstance(checkpoint["state_dict"], dict)
    ):
        sd = checkpoint["state_dict"]
    if not isinstance(sd, dict):
        raise TypeError("Unsupported checkpoint format; expected dict-like state_dict.")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=strict)
    return True




## === cell 1
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)




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
ex_csv = "../input/extra-images/extra_image.csv"
if os.path.exists(ex_csv):
    ex_df = pd.read_csv(ex_csv, index_col=0)
    ex_df["path"] = "../input/extra-images/images/images/" + ex_df["image_id"].astype(
        str
    )
    ex_df = ex_df.drop(columns=["image_id"])
else:
    ex_df = pd.DataFrame(columns=["label", "path"])
ex_df.head()




## === cell 7
ex_df




## === cell 8
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)

if len(ex_df) > 0:
    train_df = pd.concat([train_df, ex_df], ignore_index=True)




## === cell 9
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    train_df.label.value_counts().plot(kind="bar")




## === cell 10
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    valid_df.label.value_counts().plot(kind="bar")




## === cell 11
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()




## === cell 12
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()




## === cell 13
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    im = Image.open(train_df["path"][0])
    im




## === cell 14
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    im




## === cell 15
pass




## === cell 16
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image


class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        df_ = dataframe.reset_index(drop=True)
        self.paths = df_["path"].to_numpy()
        self.labels = df_["label"].to_numpy(dtype="int64", copy=False)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index])

        img_t = read_image(path)  # uint8 [C,H,W]
        image = img_t

        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 17
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        if random.random() >= self.p:
            return image

        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                return image
            _, h, w = image.shape
            ms = self.mask_size
            if w <= ms or h <= ms:
                return image
            for _ in range(10):
                x = random.randrange(0, w - ms)
                y = random.randrange(0, h - ms)
                image[:, y : y + ms, x : x + ms] = 0
            return image

        start_width, start_height = [], []
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




## === cell 18
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device




## === cell 19
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
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=res_cfg["mean"], std=res_cfg["std"]),
    ]
)


class _EnsureFloat32:
    def __call__(self, x):
        if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
            return (
                x.to(torch.float32).div(255.0)
                if x.dtype == torch.uint8
                else x.to(torch.float32)
            )
        return x


res_valid_transform = transforms.Compose(
    [_EnsureFloat32(), timm.data.create_transform(**res_cfg, is_training=False)]
)
ef_valid_transform = transforms.Compose(
    [_EnsureFloat32(), timm.data.create_transform(**ef_cfg, is_training=False)]
)

res_inference_transform = res_valid_transform
ef_inference_transform = ef_valid_transform

inference_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=res_cfg["mean"], std=res_cfg["std"]),
    ]
)




## === cell 20
dataset = CassavaDataset(train_df, train_transform)




## === cell 21
pass




## === cell 22
map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 23
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 24
unnorm = Unnormalize(res_cfg["mean"], res_cfg["std"])




## === cell 25
def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img)
    plt.imshow(img.permute(1, 2, 0))
    if label is not None:
        plt.title(label_to_name[str(label)])




## === cell 26
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




## === cell 27
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    tensor, label = dataset[0]
    display_img(tensor, unnorm, label)




## === cell 28
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    loader = DataLoader(
        dataset,
        16,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=_prefetch_factor_for(2),
    )
    display_batch(next(iter(loader)))




## === cell 29
epoch = 3
batch_size = 16




## === cell 30
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)

res_w = "../input/pretrined-models/models/models/pretrained_resNet.pth"
if os.path.exists(res_w):
    load_state_dict_flexible(resNet, res_w, strict=True)
else:
    try:
        tmp = timm.create_model("resnet50", pretrained=True)
        resNet.load_state_dict(tmp.state_dict(), strict=False)
        res_cfg = timm.data.resolve_data_config({}, model=resNet)
        res_valid_transform = transforms.Compose(
            [_EnsureFloat32(), timm.data.create_transform(**res_cfg, is_training=False)]
        )
        res_inference_transform = res_valid_transform
    except Exception as e:
        warnings.warn(
            f"Could not load timm pretrained resnet50; continuing with random init. Error: {e}",
            RuntimeWarning,
        )

resNet = resNet.to(device)




## === cell 31
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)

ef_w = "../input/pretrined-models/models/models/pretrained_ef_model.pth"
if os.path.exists(ef_w):
    load_state_dict_flexible(ef_model, ef_w, strict=True)
else:
    try:
        tmp = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
        ef_model.load_state_dict(tmp.state_dict(), strict=False)
        ef_cfg = timm.data.resolve_data_config({}, model=ef_model)
        ef_valid_transform = transforms.Compose(
            [_EnsureFloat32(), timm.data.create_transform(**ef_cfg, is_training=False)]
        )
        ef_inference_transform = ef_valid_transform
    except Exception as e:
        warnings.warn(
            f"Could not load timm pretrained tf_efficientnet_b2_ns; continuing with random init. Error: {e}",
            RuntimeWarning,
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
def calc_correction(model, df_, valid_transform_):
    model.eval()

    ds = CassavaDataset(df_, transform=valid_transform_)
    nw = _default_num_workers("valid")
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=nw,
        pin_memory=USE_CUDA,
        persistent_workers=True if nw > 0 else False,
        prefetch_factor=_prefetch_factor_for(nw),
    )

    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)
    pred_counts = torch.zeros(5, device=device, dtype=torch.long)

    with torch.inference_mode():
        for xb, yb in tqdm(loader, total=len(loader), disable=True):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).argmax(1)
            total += yb.numel()
            correct += (pred == yb).sum()
            pred_counts += torch.bincount(pred, minlength=5)

    percent = (correct.float() / total.clamp_min(1)).item()
    return percent, pred_counts.detach().cpu().tolist()




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
def _make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=USE_CUDA,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=_prefetch_factor_for(num_workers),
    )


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        print("fold: ", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = _make_loader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=_default_num_workers("train"),
        )

        valid_dataset = Subset(dataset, valid_index)
        valid_loader = _make_loader(
            valid_dataset,
            batch_size,
            shuffle=False,
            num_workers=_default_num_workers("valid"),
        )

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
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
                train_loss += loss.item() * len(data)
            train_loss = train_loss / len(train_loader.sampler)
            train_losses.append(train_loss)

            model.eval()
            with torch.inference_mode():
                for data, target in valid_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
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




## === cell 36
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
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
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
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
        plot_losses(epoch, title, train_losses, valid_losses)




## === cell 37
pass




## === cell 38
ef_local = "./ef_model.pth"
res_local = "./res_model.pth"

ef_input_candidates = [
    "../input/pretrined-models/models/models/ef_model.pth",
    "../input/pretrined-models/models/models/pretrained_ef_model.pth",
]
res_input_candidates = [
    "../input/pretrined-models/models/models/res_model.pth",
    "../input/pretrined-models/models/models/pretrained_resNet.pth",
]


def _load_if_exists(model, pth):
    if os.path.exists(pth):
        try:
            load_state_dict_flexible(model, pth, strict=True)
            return True
        except Exception as e:
            warnings.warn(
                f"Found checkpoint at {pth} but could not load it: {e}", RuntimeWarning
            )
            return False
    return False


has_ef = any(
    _load_if_exists(ef_model, p) for p in ef_input_candidates
) or _load_if_exists(ef_model, ef_local)
has_res = any(
    _load_if_exists(resNet, p) for p in res_input_candidates
) or _load_if_exists(resNet, res_local)

ef_model = ef_model.to(device)
resNet = resNet.to(device)

if USE_CUDA:
    try:
        resNet = resNet.to(memory_format=torch.channels_last)
        ef_model = ef_model.to(memory_format=torch.channels_last)
    except Exception:
        pass

try:
    if hasattr(torch, "compile"):
        resNet = torch.compile(resNet, mode="reduce-overhead")
        ef_model = torch.compile(ef_model, mode="reduce-overhead")
except Exception:
    pass




## === cell 39
def fine_tune_full_data(
    model, optimizer, scheduler, criterion, full_df, transform, epoch, batch_size
):
    full_ds = CassavaDataset(full_df, transform)
    nw = _default_num_workers("train")
    full_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=USE_CUDA,
        persistent_workers=True if nw > 0 else False,
        prefetch_factor=_prefetch_factor_for(nw),
    )
    model.train()
    for ep in range(1, epoch + 1):
        epoch_loss = 0.0
        for data, target in full_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
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


need_train_res = not has_res
need_train_ef = not has_ef

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




## === cell 40
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




## === cell 41
classifier = CassaveClassifier(resNet, ef_model).to(device)




## === cell 42
def test_rate():
    for rate in range(1, 10):
        classifier.eval()
        ds = CassavaDataset(valid_df, transform=res_valid_transform)
        nw = _default_num_workers("valid")
        loader = DataLoader(
            ds,
            batch_size=64,
            shuffle=False,
            num_workers=nw,
            pin_memory=USE_CUDA,
            persistent_workers=True if nw > 0 else False,
            prefetch_factor=_prefetch_factor_for(nw),
        )

        count = torch.zeros((), device=device, dtype=torch.long)
        total = torch.zeros((), device=device, dtype=torch.long)
        pred_counts = torch.zeros(5, device=device, dtype=torch.long)

        with torch.inference_mode():
            for xb, yb in tqdm(loader, total=len(loader), disable=True):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = classifier.test(xb, rate / 10).argmax(1)
                total += yb.numel()
                count += (pred == yb).sum()
                pred_counts += torch.bincount(pred, minlength=5)

        percent = (count.float() / total.clamp_min(1)).item()
        print("rate: ", rate / 10)
        print("percent: ", percent)




## === cell 43
pass




## === cell 44
pass




## === cell 45
test_dir = "../input/cassava-leaf-disease-classification/test_images/"




## === cell 46
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




## === cell 47
classifier.eval()
resNet.eval()
ef_model.eval()


class TestImageDatasetDual(Dataset):
    def __init__(self, paths, res_size, ef_size, res_mean, res_std, ef_mean, ef_std):
        self.paths = list(paths)
        self.res_size = int(res_size)
        self.ef_size = int(ef_size)
        self.res_mean = res_mean
        self.res_std = res_std
        self.ef_mean = ef_mean
        self.ef_std = ef_std

        self._res_tf = transforms.Compose(
            [
                transforms.Resize((self.res_size, self.res_size)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=self.res_mean, std=self.res_std),
            ]
        )
        self._ef_tf = transforms.Compose(
            [
                transforms.Resize((self.ef_size, self.ef_size)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=self.ef_mean, std=self.ef_std),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        if not os.path.isfile(pth):
            x_res = torch.zeros(3, self.res_size, self.res_size, dtype=torch.float32)
            x_ef = torch.zeros(3, self.ef_size, self.ef_size, dtype=torch.float32)
            ok = 0
            return x_res, x_ef, ok

        img_t = read_image(pth)  # uint8 tensor [C,H,W]
        x_res = self._res_tf(img_t)
        x_ef = self._ef_tf(img_t)
        ok = 1
        return x_res, x_ef, ok


res_sz = int(res_cfg.get("input_size", (3, 224, 224))[1])
ef_sz = int(ef_cfg.get("input_size", (3, 260, 260))[1])

test_ds = TestImageDatasetDual(
    image_path,
    res_size=res_sz,
    ef_size=ef_sz,
    res_mean=res_cfg["mean"],
    res_std=res_cfg["std"],
    ef_mean=ef_cfg["mean"],
    ef_std=ef_cfg["std"],
)

nw = _default_num_workers("test")
test_loader = DataLoader(
    test_ds,
    batch_size=256,  # --- Timeout fix: larger batch for fewer steps; semantics unchanged.
    shuffle=False,
    num_workers=nw,
    pin_memory=USE_CUDA,
    persistent_workers=True if nw > 0 else False,
    prefetch_factor=_prefetch_factor_for(nw),
)

pred = []
with torch.inference_mode():
    for xb_res, xb_ef, okb in tqdm(test_loader, total=len(test_loader), disable=True):
        okb = okb.to(torch.int64)

        xb_res = xb_res.to(device, non_blocking=True)
        xb_ef = xb_ef.to(device, non_blocking=True)

        if USE_CUDA:
            try:
                xb_res = xb_res.contiguous(memory_format=torch.channels_last)
                xb_ef = xb_ef.contiguous(memory_format=torch.channels_last)
            except Exception:
                pass

        logits_res = resNet(xb_res)
        logits_ef = ef_model(xb_ef)

        logits = 0.5 * logits_res + 0.5 * logits_ef
        batch_pred = logits.argmax(1).detach().cpu()

        okb = okb.detach().cpu()
        batch_pred = torch.where(okb == 1, batch_pred, torch.zeros_like(batch_pred))
        pred.extend(batch_pred.tolist())

len(pred), pred[:5]




## === cell 48
pred[:20]




## === cell 49
sub = pd.DataFrame({"image_id": image_id, "label": pred})
sub.head()




## === cell 50
sub.shape, sub.isna().sum()




## === cell 51
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
