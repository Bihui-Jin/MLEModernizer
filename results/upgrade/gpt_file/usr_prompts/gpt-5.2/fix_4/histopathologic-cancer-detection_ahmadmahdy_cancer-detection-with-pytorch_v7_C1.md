# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
seaborn==0.12.2
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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
from PIL import Image
import copy

from tqdm.notebook import tqdm

try:
    from torchsummary import summary
except Exception:
    summary = None  # keep runnability if torchsummary isn't available

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 1
path2labels = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
labels_df = pd.read_csv(path2labels)
labels_df.head()



## === cell 2
labels_df.shape



## === cell 3
labels_df["label"].value_counts()



## === cell 4
print(f"The dataset has {sum(labels_df.duplicated())} duplicates")



## === cell 5
pass




## === cell 6
class cancer_dataset(Dataset):
    """
    Fix: use a robust id->label mapping instead of labels_df.loc[filename[:-4]] which
    can fail if the index isn't set correctly or if filenames contain unexpected chars.
    Also support both train and test by making labels optional.
    """

    def __init__(self, data_dir, transform, data_type="train"):
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform

        path2data = os.path.join(data_dir, data_type)
        filenames = sorted(
            [f for f in os.listdir(path2data) if f.lower().endswith(".tif")]
        )

        self.filenames = filenames
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        self.ids = [os.path.splitext(f)[0] for f in filenames]

        if data_type == "train":
            path2labels = os.path.join(data_dir, "train_labels.csv")
            df = pd.read_csv(path2labels)
            self.id2label = dict(zip(df["id"].values, df["label"].values))
            missing = [i for i in self.ids if i not in self.id2label]
            if len(missing) > 0:
                raise KeyError(
                    f"{len(missing)} train image ids missing from train_labels.csv. Example: {missing[0]}"
                )
            self.labels = [int(self.id2label[i]) for i in self.ids]
        else:
            self.labels = None

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.full_filenames[idx]).convert("RGB")  # enforce 3-channel
        img = self.transform(img)
        if self.labels is None:
            return img, self.ids[idx]
        return img, self.labels[idx]




## === cell 7
data_transformer = transforms.Compose(
    [transforms.CenterCrop(32), transforms.Resize((46, 46)), transforms.ToTensor()]
)



## === cell 8
data_dir = "/kaggle/input/histopathologic-cancer-detection"
img_dataset = cancer_dataset(data_dir, data_transformer, "train")
len(img_dataset), img_dataset[0][0].shape, img_dataset[0][1]



## === cell 9
img, label = img_dataset[19]
print(img.shape, torch.min(img), torch.max(img), label)



## === cell 10
len_dataset = len(img_dataset)
len_train = int(0.8 * len_dataset)
len_val = len_dataset - len_train

g = torch.Generator().manual_seed(0)
train_ds, val_ds = random_split(img_dataset, [len_train, len_val], generator=g)

print(f"train dataset length: {len(train_ds)}")
print(f"validation dataset length: {len(val_ds)}")



## === cell 11
pass



## === cell 12
try:
    import torchvision.transforms.v2 as T2

    _HAS_T2 = True
except Exception:
    _HAS_T2 = False

if _HAS_T2:
    train_transf = T2.Compose(
        [
            T2.CenterCrop(32),
            T2.Resize((96, 96)),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),  # equivalent to ToTensor scaling
            T2.RandomHorizontalFlip(p=0.5),
            T2.RandomVerticalFlip(p=0.5),
            T2.RandomRotation(45),
            T2.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            T2.Resize((46, 46)),
        ]
    )

    val_transf = T2.Compose(
        [
            T2.CenterCrop(32),
            T2.Resize((46, 46)),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
        ]
    )
else:
    train_transf = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.Resize((46, 46)),
            transforms.ToTensor(),
        ]
    )

    val_transf = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.Resize((46, 46)),
            transforms.ToTensor(),
        ]
    )


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = Image.open(self.dataset.full_filenames[real_idx]).convert("RGB")
        img = self.transform(img)
        label = self.dataset.labels[real_idx]
        return img, label


train_ds_t = SubsetWithTransform(train_ds, train_transf)
val_ds_t = SubsetWithTransform(val_ds, val_transf)



## === cell 13
train_ds_t.transform



## === cell 14
num_workers = min(8, (os.cpu_count() or 2))

train_dl = DataLoader(
    train_ds_t,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
val_dl = DataLoader(
    val_ds_t,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

for x, y in train_dl:
    print("train batch:", x.shape, y.shape)
    break

for x, y in val_dl:
    print("val batch:", x.shape, y.shape)
    break




## === cell 15
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)

        self.dropout_rate = 0.25
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 1 * 64, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, X):
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 1 * 1 * 64)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


cnn_model = Network()
model = cnn_model.to(device)

if summary is not None:
    summary(cnn_model, input_size=(3, 46, 46), device=device.type)



## === cell 16
loss_func = nn.NLLLoss(reduction="sum")



## === cell 17
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)




## === cell 18
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    if opt is not None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in tqdm(dataset_dl, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        run_loss += loss_b
        t_metric += metric_b

    loss = run_loss / float(len_data)
    metric = t_metric / float(len_data)
    return loss, metric




## === cell 19
def train_val(model, params, verbose=False):
    epochs = params["epochs"]
    opt = params["optimiser"]
    loss_func = params["f_loss"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in tqdm(range(epochs), leave=False):
        current_lr = get_lr(opt)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, current lr={current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.inference_mode():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Saved best model weights")

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                print("Loading best model weights")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(
                f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}"
            )
            print("-" * 20)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history




## === cell 20
params_train = {
    "train": train_dl,
    "val": val_dl,
    "epochs": 50,
    "optimiser": opt,
    "lr_change": lr_scheduler,
    "f_loss": loss_func,
    "weight_path": "weights.pt",
}

model, loss_hist, metric_hist = train_val(model, params_train, verbose=True)



## === cell 21
import seaborn as sns

sns.set(style="whitegrid")

if "loss_hist" in globals() and "metric_hist" in globals():
    epochs = params_train["epochs"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(
        x=list(range(1, epochs + 1)),
        y=loss_hist["train"],
        ax=ax[0],
        label='loss_hist["train"]',
    )
    sns.lineplot(
        x=list(range(1, epochs + 1)),
        y=loss_hist["val"],
        ax=ax[0],
        label='loss_hist["val"]',
    )
    ax[0].set_title("Loss")

    sns.lineplot(
        x=list(range(1, epochs + 1)),
        y=metric_hist["train"],
        ax=ax[1],
        label='metric_hist["train"]',
    )
    sns.lineplot(
        x=list(range(1, epochs + 1)),
        y=metric_hist["val"],
        ax=ax[1],
        label='metric_hist["val"]',
    )
    ax[1].set_title("Accuracy")

    plt.show()
else:
    print("Skipping plots because loss_hist/metric_hist are not available.")




## === cell 22
class cancerdata_test(Dataset):
    def __init__(self, data_dir, transform, data_type="test"):
        path2data = os.path.join(data_dir, data_type)
        filenames = sorted(
            [f for f in os.listdir(path2data) if f.lower().endswith(".tif")]
        )
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        self.ids = [os.path.splitext(f)[0] for f in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.ids[idx]




## === cell 23
if os.path.exists("weights.pt"):
    model.load_state_dict(torch.load("weights.pt", map_location=device))
model.eval()



## === cell 24
path2sub = "/kaggle/input/histopathologic-cancer-detection/sample_submission.csv"
sample_sub = pd.read_csv(path2sub)

data_dir = "/kaggle/input/histopathologic-cancer-detection/"
if _HAS_T2:
    test_transform = T2.Compose(
        [
            T2.CenterCrop(32),
            T2.Resize((46, 46)),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
        ]
    )
else:
    test_transform = transforms.Compose(
        [transforms.CenterCrop(32), transforms.Resize((46, 46)), transforms.ToTensor()]
    )

img_dataset_test = cancerdata_test(data_dir, test_transform, data_type="test")
print(len(img_dataset_test), "test samples found")




## === cell 25
def inference_proba(model, dataset, device, batch_size=64):
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    all_ids = []
    all_p1 = []

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for xb, ids in tqdm(dl, leave=False):
            xb = xb.to(device, non_blocking=True)
            logp = model(xb)  # log-softmax over 2 classes
            p = torch.exp(logp)  # probabilities
            p1 = p[:, 1].detach().cpu().numpy()
            all_p1.append(p1)
            all_ids.extend(list(ids))

    all_p1 = np.concatenate(all_p1, axis=0)
    return all_ids, all_p1


test_ids, test_p1 = inference_proba(model, img_dataset_test, device)

pred_df = pd.DataFrame({"id": test_ids, "label": test_p1})

submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")
if submission["label"].isna().any():
    raise RuntimeError("Some test ids did not get predictions; submission has NaNs.")

submission.head()



## === cell 26
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
