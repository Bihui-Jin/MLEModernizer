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
import time
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
from PIL import Image

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_on_gpu = torch.cuda.is_available()
print(device)



## === cell 1
path2labels = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
labels_df = pd.read_csv(path2labels)



## === cell 2
print(labels_df.head())
print(labels_df.shape)



## === cell 3
print(labels_df["label"].value_counts())



## === cell 4
print(f"The dataset has {int(labels_df.duplicated().sum())} duplicates")



## === cell 5
pass




## === cell 6
class cancer_dataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        path2data = os.path.join(data_dir, data_type)

        path2labels = os.path.join(data_dir, "train_labels.csv")
        labels_df = pd.read_csv(path2labels)
        labels_df.set_index("id", inplace=True)

        all_entries = os.listdir(path2data)
        filenames = [
            f
            for f in all_entries
            if f.lower().endswith(".tif") and (f[:-4] in labels_df.index)
        ]

        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        self.labels = [int(labels_df.loc[fn[:-4], "label"]) for fn in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.full_filenames[idx])
        img = self.transform(img)
        return img, self.labels[idx]




## === cell 7
data_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((46, 46)),
    ]
)



## === cell 8
pass



## === cell 9
data_dir = "/kaggle/input/histopathologic-cancer-detection"
img_dataset = cancer_dataset(
    data_dir=data_dir, transform=data_transformer, data_type="train"
)

img, label = img_dataset[19]
print(img.shape, torch.min(img), torch.max(img))



## === cell 10
len_dataset = len(img_dataset)
len_train = int(0.8 * len_dataset)
len_val = len_dataset - len_train

g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(img_dataset, [len_train, len_val], generator=g)

print(f"train dataset length: {len(train_ds)}")
print(f"validation dataset length: {len(val_ds)}")



## === cell 11
pass



## === cell 12
train_transf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
    ]
)

val_transf = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_ds.dataset.transform = train_transf
val_ds.dataset.transform = val_transf



## === cell 13
print(train_ds.dataset.transform)



## === cell 14
num_workers = min(8, (os.cpu_count() or 2))
pin_memory = torch.cuda.is_available()

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    generator=torch.Generator().manual_seed(SEED),
)

val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)
xb, yb = next(iter(val_dl))
print(xb.shape, yb.shape)




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
        x = F.dropout(x, self.dropout_rate)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


cnn_model = Network()
model = cnn_model.to(device)

print(model)



## === cell 16
loss_func = nn.NLLLoss(reduction="sum")



## === cell 17
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, verbose=0)




## === cell 18
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    if opt is not None:
        opt.zero_grad(set_to_none=True)  # speed, equivalent gradients
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
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

    for epoch in range(epochs):
        current_lr = get_lr(opt)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, current lr={current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
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
                f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}"
            )
            print("-" * 20)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history




## === cell 20
def _patched_forward(self, X):
    x = self.pool(F.relu(self.conv1(X)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))

    x = x.view(x.size(0), -1)  # keep batch size intact

    if self.fc1.in_features != x.size(1):
        self.fc1 = nn.Linear(x.size(1), self.fc1.out_features).to(x.device)

    x = F.relu(self.fc1(x))
    x = F.dropout(x, self.dropout_rate, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


Network.forward = _patched_forward

params_train = {
    "train": train_dl,
    "val": val_dl,
    "epochs": 50,
    "optimiser": opt,
    "lr_change": lr_scheduler,
    "f_loss": loss_func,
    "weight_path": "weights.pt",
}

t0 = time.time()
model, loss_hist, metric_hist = train_val(model, params_train, verbose=True)
print(f"Training time (s): {time.time() - t0:.1f}")


## === cell 21
pass




## === cell 22
class cancerdata_test(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        path2data = os.path.join(data_dir, data_type)
        filenames = [f for f in os.listdir(path2data) if f.lower().endswith(".tif")]
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]

        csv_filename = "sample_submission.csv"
        path2csvLabels = os.path.join(data_dir, csv_filename)
        labels_df = pd.read_csv(path2csvLabels)
        labels_df.set_index("id", inplace=True)

        self.labels = [int(labels_df.loc[fn[:-4]].values[0]) for fn in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]




## === cell 23
model.load_state_dict(torch.load("weights.pt", map_location=device))



## === cell 24
path2sub = "/kaggle/input/histopathologic-cancer-detection/sample_submission.csv"
labels_df = pd.read_csv(path2sub)
data_dir = "/kaggle/input/histopathologic-cancer-detection/"

data_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((46, 46)),
    ]
)

img_dataset_test = cancerdata_test(data_dir, data_transformer, data_type="test")
print(len(img_dataset_test), "samples found")




## === cell 25
def inference(model, dataset, device, num_classes=2, batch_size=256):
    model = model.to(device)
    model.eval()

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, (os.cpu_count() or 2)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(min(8, (os.cpu_count() or 2)) > 0),
        prefetch_factor=4 if (min(8, (os.cpu_count() or 2)) > 0) else None,
    )

    len_data = len(dataset)
    y_out = torch.empty((len_data, num_classes), dtype=torch.float32)  # CPU
    y_gt = np.zeros((len_data,), dtype="uint8")

    offset = 0
    with torch.no_grad():
        for xb, yb in dl:
            bs = xb.size(0)
            y_gt[offset : offset + bs] = yb.numpy()
            out = model(xb.to(device, non_blocking=True)).detach().cpu()
            y_out[offset : offset + bs] = out
            offset += bs

    return y_out.numpy(), y_gt


t0 = time.time()
y_test_out, _ = inference(model, img_dataset_test, device, batch_size=256)
print(f"Inference time (s): {time.time() - t0:.1f}")

y_test_pred = np.argmax(y_test_out, axis=1)

test_ids = [
    os.path.basename(name).split(".")[0] for name in img_dataset_test.full_filenames
]
test_preds = pd.DataFrame({"img": test_ids, "preds": y_test_pred})
submission = pd.merge(labels_df, test_preds, left_on="id", right_on="img")
submission = submission[["id", "preds"]]
submission.columns = ["id", "label"]
print(submission.head())



## === cell 26
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(submission), "rows")
