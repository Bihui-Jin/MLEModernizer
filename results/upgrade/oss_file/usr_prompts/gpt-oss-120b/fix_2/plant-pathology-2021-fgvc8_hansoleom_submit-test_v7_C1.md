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

# 5. Target score

0.7137396121883658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
import os
import torch
import torchvision
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler
from efficientnet_pytorch import (
    EfficientNet,
)  # placeholder, will be replaced below if import fails



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/966514775.py in <cell line: 0>()
      9 from torch import optim
     10 from torch.optim import lr_scheduler
---> 11 from efficientnet_pytorch import (
     12     EfficientNet,
     13 )  # placeholder, will be replaced below if import fails

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
with open(csv_path, "r") as f:
    lines = f.readlines()[1:]  # skip header
import random

for _ in range(5):
    random.shuffle(lines)
cnt = int(len(lines) * 0.9)
train_csv = lines[:cnt]
valid_csv = lines[cnt:]



## === cell 3
label2idx = {
    "frog_eye_leaf_spot": 0,
    "rust frog_eye_leaf_spot": 1,
    "healthy": 2,
    "scab frog_eye_leaf_spot complex": 3,
    "rust": 4,
    "powdery_mildew complex": 5,
    "frog_eye_leaf_spot complex": 6,
    "powdery_mildew": 7,
    "rust complex": 8,
    "scab frog_eye_leaf_spot": 9,
    "scab": 10,
    "complex": 11,
}
idx2label = {v: k for k, v in label2idx.items()}


class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv_lines, transform=None):
        self.csv = csv_lines
        self.root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        line = self.csv[idx].strip()
        image_name, label_name = line.split(",")
        img = Image.open(os.path.join(self.root, image_name)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_idx = label2idx[label_name.strip()]
        return img, label_idx




## === cell 4
train_dataset = torchvision_Dataset("train_images", train_csv, transform_train)
valid_dataset = torchvision_Dataset("train_images", valid_csv, transform_valid)

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=1
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=1
)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 6
try:
    from efficientnet_pytorch import EfficientNet

    model_ft = EfficientNet.from_name("efficientnet-b4", num_classes=len(label2idx))
except Exception:
    model_ft = torchvision.models.efficientnet_b4(weights=None)
    in_features = model_ft.classifier[1].in_features
    model_ft.classifier[1] = torch.nn.Linear(in_features, len(label2idx))
model_ft = model_ft.to(device)



## === cell 7
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=30, eta_min=0
)


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)


exp_lr_scheduler = GradualWarmupScheduler(
    optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler
)




## === cell 8
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_corrects = 0
        train_cnt = 0
        for inputs, labels in tqdm(
            train_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} - Train"
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            train_cnt += inputs.size(0)
        scheduler.step()
        model.eval()
        valid_corrects = 0
        valid_cnt = 0
        for inputs, labels in tqdm(
            valid_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} - Valid"
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            valid_corrects += torch.sum(preds == labels.data)
            valid_cnt += inputs.size(0)
        epoch_acc = valid_corrects.double() / valid_cnt
        print(f"Epoch {epoch+1} validation accuracy: {epoch_acc:.4f}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/best_epoch_{epoch}.pth")
    model.load_state_dict(best_model_wts)
    return model




## === cell 9
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3165070179.py in <cell line: 0>()
      1 # Train briefly (optional – can be skipped to reduce runtime)
----> 2 model_ft = train_model(
      3     model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2
      4 )
      5 

/tmp/ipykernel_55/673015680.py in train_model(model, criterion, optimizer, scheduler, num_epochs)
      7         train_corrects = 0
      8         train_cnt = 0
----> 9         for inputs, labels in tqdm(
     10             train_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} - Train"
     11         ):

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3547993179.py", line 31, in __getitem__
    img = Image.open(os.path.join(self.root, image_name)).convert("RGB")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'train_images/885e1bc6a3674eb1.jpg'


## === cell 10
ckpt_path = "/kaggle/input/best-model/28.pth"
if os.path.exists(ckpt_path):
    model_ft.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_ft.to(device)
else:
    print("Checkpoint not found – using the trained model above.")



## === cell 11
from glob import glob
import pandas as pd

test_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
img_paths = glob(os.path.join(test_img_dir, "*"))
submission_rows = []

model_ft.eval()
with torch.no_grad():
    for img_path in tqdm(img_paths, desc="Generating predictions"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform_valid(img).unsqueeze(0).to(device)
        outputs = model_ft(img_tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_label = idx2label[pred_idx.item()]
        filename = os.path.basename(img_path)
        submission_rows.append([filename, pred_label])

submission_df = pd.DataFrame(submission_rows, columns=["image", "labels"])
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/736689536.py in <cell line: 0>()
      9 with torch.no_grad():
     10     for img_path in tqdm(img_paths, desc="Generating predictions"):
---> 11         img = Image.open(img_path).convert("RGB")
     12         img_tensor = transform_valid(img).unsqueeze(0).to(device)
     13         outputs = model_ft(img_tensor)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/plant-pathology-2021-fgvc8/test_images/test_images'
