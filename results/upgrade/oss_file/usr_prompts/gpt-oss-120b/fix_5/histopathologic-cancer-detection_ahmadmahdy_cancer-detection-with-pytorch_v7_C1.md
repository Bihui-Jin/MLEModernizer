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
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchsummary import summary
import copy
from PIL import Image  # added import for image loading

torch.manual_seed(0)




## === cell 1
train_labels_path = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
labels_df = pd.read_csv(train_labels_path)




## === cell 2
class CancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        self.transform = transform
        data_path = os.path.join(data_dir, data_type)
        self.filenames = [
            f for f in sorted(os.listdir(data_path)) if f.lower().endswith(".tif")
        ]
        self.full_paths = [os.path.join(data_path, f) for f in self.filenames]

        labels_path = os.path.join(data_dir, "train_labels.csv")
        df = pd.read_csv(labels_path)
        df.set_index("id", inplace=True)
        self.labels = [
            (
                int(df.loc[os.path.splitext(fname)[0]].values[0])
                if os.path.splitext(fname)[0] in df.index
                else 0
            )
            for fname in self.filenames
        ]

    def __len__(self):
        return len(self.full_paths)

    def __getitem__(self, idx):
        img = Image.open(self.full_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label




## === cell 3
data_transform = transforms.Compose(
    [
        transforms.Resize((46, 46)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ]
)




## === cell 4
data_dir = "/kaggle/input/histopathologic-cancer-detection"
full_dataset = CancerDataset(data_dir, data_transform, data_type="train")




## === cell 5
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])




## === cell 6
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)




## === cell 7
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)  # -> 30x30
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # -> 13x13 after pooling
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)  # -> 4x4 after pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 2)
        self.dropout_rate = 0.25

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 30 -> 15
        x = self.pool(F.relu(self.conv2(x)))  # 13 -> 6
        x = self.pool(F.relu(self.conv3(x)))  # 4 -> 2
        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
print("Using device:", device)
summary(model, input_size=(3, 32, 32), device=device.type)




## === cell 9
criterion = nn.NLLLoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=False
)


def get_lr(opt):
    for pg in opt.param_groups:
        return pg["lr"]


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), correct


def loss_epoch(model, loss_func, data_loader, opt=None):
    model.train() if opt else model.eval()
    total_loss = 0.0
    total_correct = 0
    n_samples = 0
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        output = model(xb)
        loss, correct = loss_batch(loss_func, output, yb, opt)
        total_loss += loss * xb.size(0)
        total_correct += correct
        n_samples += xb.size(0)
    return total_loss / n_samples, total_correct / n_samples




## === cell 10
def train_val(model, params, verbose=False):
    epochs = params["epochs"]
    opt = params["optimiser"]
    loss_func = params["loss_func"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_sched = params["lr_sched"]
    weight_path = params["weight_path"]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} – LR: {get_lr(opt):.6f}")
        train_loss, train_acc = loss_epoch(model, loss_func, train_dl, opt)
        val_loss, val_acc = loss_epoch(model, loss_func, val_dl, None)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, weight_path)

        lr_sched.step(val_loss)

        if verbose:
            print(
                f"  train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
                f"train acc: {train_acc*100:.2f}%, val acc: {val_acc*100:.2f}%"
            )
    model.load_state_dict(best_model_wts)
    return model




## === cell 11
train_params = {
    "train": train_loader,
    "val": val_loader,
    "epochs": 3,  # a few epochs to get reasonable performance
    "optimiser": optimizer,
    "lr_sched": scheduler,
    "loss_func": criterion,
    "weight_path": "best_weights.pt",
}
model = train_val(model, train_params, verbose=True)




## === cell 12
class CancerTestDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        test_path = os.path.join(data_dir, "test")
        self.filenames = sorted(
            [f for f in os.listdir(test_path) if f.lower().endswith(".tif")]
        )
        self.full_paths = [os.path.join(test_path, f) for f in self.filenames]
        self.ids = [os.path.splitext(f)[0] for f in self.filenames]

    def __len__(self):
        return len(self.full_paths)

    def __getitem__(self, idx):
        img = Image.open(self.full_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, 0




## === cell 13
test_dataset = CancerTestDataset(data_dir, data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)




## === cell 14
def inference(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Inference"):
            xb = xb.to(device)
            logp = model(xb)  # log‑softmax
            p = torch.exp(logp)[:, 1]  # probability of class 1
            probs.extend(p.cpu().numpy())
    return np.array(probs)


test_probs = inference(model, test_loader, device)




## === cell 15
submission = pd.DataFrame({"id": test_dataset.ids, "label": test_probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")
