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

# 5. Target score

0.71898430286242

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.models import ResNeXt50_32X4D_Weights

from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 1
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, images, labels=None):
        if self.training:
            logits = self.backbone(images)
            loss = self.loss(logits, labels)
            return loss
        else:
            logits = self.backbone(images)
            probs = torch.sigmoid(logits)
            batched_labels = [torch.where(p > 0.5)[0].tolist() for p in probs]
            return batched_labels




## === cell 2
class TestDataset:
    def __init__(self, root):
        self.root = root
        self.label_map = {
            "complex": ["疑难复杂", 0],
            "rust": ["锈菌，生锈", 1],
            "scab": ["疮痂病，斑点病", 2],
            "frog_eye_leaf_spot": ["青蛙眼叶斑", 3],
            "healthy": ["健康的", 4],
            "powdery_mildew": ["白粉病", 5],
        }
        self.index_to_name = {v[1]: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.files = [
            (os.path.join(dirname, fname), fname)
            for dirname, _, filenames in os.walk(self.root)
            for fname in filenames
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.trans = T.Compose(
            [
                T.Resize((300, 300)),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=1.0),
            ]
        )

    def __getitem__(self, idx):
        file_path, name = self.files[idx]
        img = io.read_image(file_path)  # fast tensor loading
        img = self.trans(img)
        return img, name

    def __len__(self):
        return len(self.files)


class TrainDataset:
    def __init__(self, csv_path, img_root, label_map):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.mlb = MultiLabelBinarizer(classes=list(label_map.keys()))
        self.mlb.fit([list(label_map.keys())])  # ensure ordering
        self.trans = T.Compose(
            [
                T.Resize((300, 300)),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=1.0),
            ]
        )
        self.targets = self.mlb.transform(
            self.df["labels"].apply(lambda x: x.split()).tolist()
        )
        self.file_names = self.df["image"].tolist()

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        file_path = os.path.join(self.img_root, img_name)
        img = io.read_image(file_path)  # fast tensor loading
        img = self.trans(img)
        label = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.file_names)




## === cell 3
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512

train_csv = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_root = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_root = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"

label_map = {
    "complex": ["疑难复杂", 0],
    "rust": ["锈菌，生锈", 1],
    "scab": ["疮痂病，斑点病", 2],
    "frog_eye_leaf_spot": ["青蛙眼叶斑", 3],
    "healthy": ["健康的", 4],
    "powdery_mildew": ["白粉病", 5],
}

train_dataset = TrainDataset(train_csv, train_root, label_map)

num_workers = min(8, os.cpu_count() or 1)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
    prefetch_factor=2,  # modest prefetch; higher values give diminishing returns
)

model = Model(num_classes=len(label_map)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()
model.train()
for epoch in range(2):  # two epochs are enough for a modest boost
    epoch_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            loss = model(imgs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {epoch_loss/len(train_loader):.4f}")

test_dataset = TestDataset(test_root)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
    prefetch_factor=2,
)

model.eval()
all_predict = []
with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            batch_labels = model(images)  # returns list of index lists
        for name, label_idxs in zip(names, batch_labels):
            label_names = [test_dataset.index_to_name[i] for i in label_idxs]
            all_predict.append([name, " ".join(label_names)])

submission = pd.DataFrame(all_predict, columns=["image", "labels"])
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(submission), "rows.")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3046447361.py in <cell line: 0>()
     46         optimizer.zero_grad()
     47         with autocast():
---> 48             loss = model(imgs, targets)
     49         scaler.scale(loss).backward()
     50         scaler.step(optimizer)

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

/tmp/ipykernel_55/2201876949.py in forward(self, images, labels)
     10     def forward(self, images, labels=None):
     11         if self.training:
---> 12             logits = self.backbone(images)
     13             loss = self.loss(logits, labels)
     14             return loss

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    283 
    284     def forward(self, x: Tensor) -> Tensor:
--> 285         return self._forward_impl(x)
    286 
    287 

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in _forward_impl(self, x)
    273         x = self.layer1(x)
    274         x = self.layer2(x)
--> 275         x = self.layer3(x)
    276         x = self.layer4(x)
    277 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    153 
    154         out = self.conv3(out)
--> 155         out = self.bn3(out)
    156 
    157         if self.downsample is not None:

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/batchnorm.py in forward(self, input)
    191         used for normalization (i.e. in eval mode when buffers are not None).
    192         """
--> 193         return F.batch_norm(
    194             input,
    195             # If buffers are not to be tracked, ensure that they won't be updated

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
   2820         _verify_batch_size(input.size())
   2821 
-> 2822     return torch.batch_norm(
   2823         input,
   2824         weight,

OutOfMemoryError: CUDA out of memory. Tried to allocate 362.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 94.88 MiB is free. Process 939959 has 47.42 GiB memory in use. Of the allocated memory 46.76 GiB is allocated by PyTorch, and 355.84 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
