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

# 5. Target score

0.7612

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 1 * 64, 100)
        self.fc2 = nn.Linear(100, 2)
        self.dropout_rate = 0.25

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 1 * 1 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
print("Using device:", device)
summary(model, input_size=(3, 32, 32), device=device.type)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/990639010.py in <cell line: 0>()
      2 model = Network().to(device)
      3 print("Using device:", device)
----> 4 summary(model, input_size=(3, 32, 32), device=device.type)
      5 
      6 

/usr/local/lib/python3.11/dist-packages/torchsummary/torchsummary.py in summary(model, input_size, batch_size, device)
     70     # make a forward pass
     71     # print(x.shape)
---> 72     model(*x)
     73 
     74     # remove these hooks

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3773927369.py in forward(self, x)
     15         x = self.pool(F.relu(self.conv2(x)))
     16         x = self.pool(F.relu(self.conv3(x)))
---> 17         x = self.pool(F.relu(self.conv4(x)))
     18         x = x.view(-1, 1 * 1 * 64)
     19         x = F.relu(self.fc1(x))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1843 
   1844         try:
-> 1845             return inner()
   1846         except Exception:
   1847             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1791                 args = bw_hook.setup_input_hook(args)
   1792 
-> 1793             result = forward_call(*args, **kwargs)
   1794             if _global_forward_hooks or self._forward_hooks:
   1795                 for hook_id, hook in (

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size

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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1761614936.py in <cell line: 0>()
      8     "weight_path": "best_weights.pt",
      9 }
---> 10 model = train_val(model, train_params, verbose=True)
     11 
     12 

/tmp/ipykernel_55/972057014.py in train_val(model, params, verbose)
     14         if verbose:
     15             print(f"Epoch {epoch+1}/{epochs} – LR: {get_lr(opt):.6f}")
---> 16         train_loss, train_acc = loss_epoch(model, loss_func, train_dl, opt)
     17         val_loss, val_acc = loss_epoch(model, loss_func, val_dl, None)
     18 

/tmp/ipykernel_55/2048995525.py in loss_epoch(model, loss_func, data_loader, opt)
     29     for xb, yb in data_loader:
     30         xb, yb = xb.to(device), yb.to(device)
---> 31         output = model(xb)
     32         loss, correct = loss_batch(loss_func, output, yb, opt)
     33         total_loss += loss * xb.size(0)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3773927369.py in forward(self, x)
     15         x = self.pool(F.relu(self.conv2(x)))
     16         x = self.pool(F.relu(self.conv3(x)))
---> 17         x = self.pool(F.relu(self.conv4(x)))
     18         x = x.view(-1, 1 * 1 * 64)
     19         x = F.relu(self.fc1(x))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1843 
   1844         try:
-> 1845             return inner()
   1846         except Exception:
   1847             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1791                 args = bw_hook.setup_input_hook(args)
   1792 
-> 1793             result = forward_call(*args, **kwargs)
   1794             if _global_forward_hooks or self._forward_hooks:
   1795                 for hook_id, hook in (

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size

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




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1089592423.py in <cell line: 0>()
     11 
     12 
---> 13 test_probs = inference(model, test_loader, device)
     14 
     15 

/tmp/ipykernel_55/1089592423.py in inference(model, loader, device)
      5         for xb, _ in tqdm(loader, desc="Inference"):
      6             xb = xb.to(device)
----> 7             logp = model(xb)  # log‑softmax
      8             p = torch.exp(logp)[:, 1]  # probability of class 1
      9             probs.extend(p.cpu().numpy())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3773927369.py in forward(self, x)
     15         x = self.pool(F.relu(self.conv2(x)))
     16         x = self.pool(F.relu(self.conv3(x)))
---> 17         x = self.pool(F.relu(self.conv4(x)))
     18         x = x.view(-1, 1 * 1 * 64)
     19         x = F.relu(self.fc1(x))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1843 
   1844         try:
-> 1845             return inner()
   1846         except Exception:
   1847             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1791                 args = bw_hook.setup_input_hook(args)
   1792 
-> 1793             result = forward_call(*args, **kwargs)
   1794             if _global_forward_hooks or self._forward_hooks:
   1795                 for hook_id, hook in (

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size

## === cell 15
submission = pd.DataFrame({"id": test_dataset.ids, "label": test_probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/902908617.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_dataset.ids, "label": test_probs})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}, shape: {submission.shape}")

NameError: name 'test_probs' is not defined
