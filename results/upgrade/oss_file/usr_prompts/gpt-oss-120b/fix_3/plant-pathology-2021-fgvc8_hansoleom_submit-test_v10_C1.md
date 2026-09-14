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

0.7626592797783938

# 6. Current score

0.28162

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21476) has done: 'I fixed the import of EfficientNet by using the torchvision implementation with pretrained ImageNet weights, corrected the label handling (avoiding variable shadowing), fixed the dataset class to return proper tensors, adjusted the model’s classifier to match the number of classes, and made the inference code use the defined device. These changes resolve the module‑not‑found errors and ensure a valid `submission.csv` is written, while keeping the original training logic unchanged.'
- What this solution (achieved 0.28162) has done: 'I modify the dataset to output multi‑hot label vectors, adjust the training loop to use those vectors directly with the focal loss, create the missing output directory, train the model for a few epochs, and change the inference step to apply a sigmoid + threshold and output space‑delimited multi‑label predictions. These targeted changes keep the original EfficientNet architecture while fixing the multi‑label handling, which should raise the F1 score toward the target.'

# 9. Code solution

## === cell 0
import os
import copy
import random
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torchvision.transforms import functional as F
from tqdm import tqdm
from PIL import Image



## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomAffine(
            degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
with open("../input/plant-pathology-2021-fgvc8/train.csv", "r") as f:
    csv_lines = f.readlines()[1:]  # skip header
    for _ in range(5):
        random.shuffle(csv_lines)
    split_idx = int(len(csv_lines) * 0.9)
    train_csv = csv_lines[:split_idx]
    valid_csv = csv_lines[split_idx:]



## === cell 3
label_set = set()
for line in train_csv:
    _, labels = line.split(",", 1)
    for lbl in labels.strip().split():
        label_set.add(lbl)
label_list = sorted(label_set)
label2idx = {lbl: idx for idx, lbl in enumerate(label_list)}




## === cell 4
class TorchVisionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_data, label_map, transform=None):
        self.csv_data = csv_data
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.num_classes = len(label_map)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        line = self.csv_data[idx].strip()
        image_name, label_names = line.split(",", 1)
        img_path = os.path.join(self.root_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = F.to_tensor(img)

        target = torch.zeros(self.num_classes, dtype=torch.float)
        for lbl in label_names.strip().split():
            if lbl in self.label_map:  # safety
                target[self.label_map[lbl]] = 1.0
        return img, target




## === cell 5
train_dataset = TorchVisionDataset(
    "train_images", train_csv, label2idx, transform=transform_train
)
valid_dataset = TorchVisionDataset(
    "train_images", valid_csv, label2idx, transform=transform_valid
)



## === cell 6
train_dataloaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=1
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=1
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 8
model_ft = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(num_ftrs, len(label2idx))
model_ft = model_ft.to(device)




## === cell 9
class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-12

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
        ce_loss = -(target * torch.log(probs) + (1 - target) * torch.log(1 - probs))
        pt = torch.where(target == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss
        return loss.mean()


criterion = FocalLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=40, eta_min=1e-6, verbose=True
)




## === cell 10
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_samples = 0

        prog_bar = tqdm(train_dataloaders, leave=False)
        for inputs, targets in prog_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            prog_bar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/train_samples:.4f}"
            )

        scheduler.step()

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in valid_dataloaders:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        preds_tensor = torch.cat(all_preds)
        targets_tensor = torch.cat(all_targets)
        tp = (preds_tensor * targets_tensor).sum()
        fp = (preds_tensor * (1 - targets_tensor)).sum()
        fn = ((1 - preds_tensor) * targets_tensor).sum()
        micro_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)

        if micro_f1 > best_f1:
            best_f1 = micro_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs("outputs", exist_ok=True)
            torch.save(best_model_wts, "outputs/best_model.pth")
            print(f"New best epoch: {epoch} with F1 {best_f1:.4f}")

    model.load_state_dict(best_model_wts)
    return model




## === cell 11
os.makedirs("outputs", exist_ok=True)



## === cell 12
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1836251218.py in <cell line: 0>()
      1 # Train the model
----> 2 model_ft = train_model(
      3     model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10
      4 )
      5 

/tmp/ipykernel_55/4290306370.py in train_model(model, criterion, optimizer, scheduler, num_epochs)
      9 
     10         prog_bar = tqdm(train_dataloaders, leave=False)
---> 11         for inputs, targets in prog_bar:
     12             inputs = inputs.to(device)
     13             targets = targets.to(device)

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
  File "/tmp/ipykernel_55/2290945640.py", line 16, in __getitem__
    img = Image.open(img_path).convert("RGB")
          ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'train_images/9b6a318cc5721d73.jpg'


## === cell 13
idx2label = {idx: lbl for lbl, idx in label2idx.items()}



## === cell 14
test_img_paths = glob("../input/plant-pathology-2021-fgvc8/test_images/*.jpg")
submission_rows = []

model_ft.eval()
with torch.no_grad():
    for img_path in tqdm(test_img_paths, desc="Predicting"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform_valid(img).unsqueeze(0).to(device)
        logits = model_ft(img_tensor)
        probs = torch.sigmoid(logits).squeeze(0)
        pred_indices = (probs > 0.5).nonzero(as_tuple=False).flatten().tolist()
        if pred_indices:
            pred_labels = [idx2label[i] for i in pred_indices]
        else:
            pred_labels = ["healthy"]
        pred_str = " ".join(pred_labels)
        img_name = os.path.basename(img_path)
        submission_rows.append([img_name, pred_str])

submission_df = pd.DataFrame(submission_rows, columns=["image", "labels"])
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
