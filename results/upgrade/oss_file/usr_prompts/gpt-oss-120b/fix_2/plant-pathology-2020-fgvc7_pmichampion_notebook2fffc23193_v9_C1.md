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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.5106922822571176

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.utils.data as Data
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import softmax
from tqdm.notebook import tqdm
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    OneOf,
    Sharpen,
    Blur,
    Resize,
    Normalize,
)
from albumentations.pytorch import ToTensorV2



## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"


def get_img_path(img_id):
    """Return absolute path to an image given its id."""
    return os.path.join(BASE_PATH, "images", f"{img_id}.jpg")




## === cell 2
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))



## === cell 3
train["image_path"] = train["image_id"].apply(get_img_path)
test["image_path"] = test["image_id"].apply(get_img_path)



## === cell 4
train_targets = train.loc[:, "healthy":"scab"]
train_paths = train["image_path"]
test_paths = test["image_path"]



## === cell 5
train_paths, valid_paths, train_targets, valid_targets = train_test_split(
    train_paths,
    train_targets,
    test_size=0.2,
    random_state=27,
    shuffle=True,
    stratify=train_targets,
)




## === cell 6
class Leaf_Dataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, test=False):
        self.paths = list(image_paths)  # list of file paths
        self.test = test
        self.labels = labels

        self.train_transform = Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                RandomBrightnessContrast(
                    p=0.7, brightness_limit=0.2, contrast_limit=0.2
                ),
                OneOf([Sharpen(p=1), Blur(p=1)], p=0.5),
                Resize(height=224, width=224),
            ]
        )

        self.eval_transform = Compose([Resize(height=224, width=224)])

        self.normalizer = Compose(
            [
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.test:
            label = torch.tensor(
                np.argmax(self.labels.iloc[idx].values.astype(int)), dtype=torch.long
            )

        if not self.test and self.labels is not None:
            img = self.train_transform(image=img)["image"]
        else:
            img = self.eval_transform(image=img)["image"]

        img = self.normalizer(image=img)["image"]

        if self.test:
            return img
        return img, label




## === cell 7
class CFG:
    batch_size = 8
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 8e-4




## === cell 8
train_dataset = Leaf_Dataset(train_paths, labels=train_targets, test=False)
valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, test=False)
test_dataset = Leaf_Dataset(test_paths, test=True)

train_loader = Data.DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
valid_loader = Data.DataLoader(
    valid_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
test_loader = Data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)



## === cell 9
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 4, bias=True),
)
model = model.to(CFG.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss()




## === cell 10
def train_one_epoch(net, loader):
    net.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, lbls in pbar:
        imgs = imgs.to(CFG.device)
        lbls = lbls.to(CFG.device)

        optimizer.zero_grad()
        outs = net(imgs)
        loss = loss_fn(outs, lbls)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outs, dim=1)
        all_labels.append(lbls.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    epoch_loss /= len(loader.dataset)
    accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return epoch_loss, accuracy


def valid_one_epoch(net, loader):
    net.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for imgs, lbls in pbar:
            imgs = imgs.to(CFG.device)
            lbls = lbls.to(CFG.device)

            outs = net(imgs)
            loss = loss_fn(outs, lbls)

            epoch_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outs, dim=1)
            all_labels.append(lbls.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    epoch_loss /= len(loader.dataset)
    accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return epoch_loss, accuracy




## === cell 11
best_val_loss = float("inf")
for epoch in range(CFG.num_epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = valid_one_epoch(model, valid_loader)
    print(
        f"Epoch {epoch+1}/{CFG.num_epochs} | "
        f"Train loss: {tr_loss:.4f}, acc: {tr_acc:.4f} | "
        f"Valid loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2417045891.py in <cell line: 0>()
      1 best_val_loss = float("inf")
      2 for epoch in range(CFG.num_epochs):
----> 3     tr_loss, tr_acc = train_one_epoch(model, train_loader)
      4     val_loss, val_acc = valid_one_epoch(model, valid_loader)
      5     print(

/tmp/ipykernel_55/1530356745.py in train_one_epoch(net, loader)
      5     all_preds = []
      6     pbar = tqdm(loader, desc="Training", leave=False)
----> 7     for imgs, lbls in pbar:
      8         imgs = imgs.to(CFG.device)
      9         lbls = lbls.to(CFG.device)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

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
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/417777012.py in __getitem__(self, idx)
     38 
     39     def __getitem__(self, idx):
---> 40         img = cv2.imread(self.paths[idx])
     41         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     42 

NameError: name 'cv2' is not defined

## === cell 12
model.load_state_dict(
    torch.load("/kaggle/working/best_model.pt", map_location=CFG.device)
)
model.eval()

all_preds = []
pbar = tqdm(test_loader, desc="Test inference", leave=False)
with torch.no_grad():
    for imgs in pbar:
        imgs = imgs.to(CFG.device)
        outs = model(imgs)  # raw logits
        probs = softmax(outs.cpu().numpy(), axis=1)
        all_preds.append(probs)

test_predictions = np.concatenate(all_preds, axis=0)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1761240586.py in <cell line: 0>()
      1 # Load best weights (in case the last epoch was not the best)
      2 model.load_state_dict(
----> 3     torch.load("/kaggle/working/best_model.pt", map_location=CFG.device)
      4 )
      5 model.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/best_model.pt'

## === cell 13
submission = pd.DataFrame(
    test_predictions, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission.insert(0, "image_id", test["image_id"])
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1602349569.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     test_predictions, columns=["healthy", "multiple_diseases", "rust", "scab"]
      3 )
      4 submission.insert(0, "image_id", test["image_id"])
      5 submission_path = "/kaggle/working/submission.csv"

NameError: name 'test_predictions' is not defined
