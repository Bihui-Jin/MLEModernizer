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

albumentations==2.0.8
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
scipy==1.15.3
seaborn==0.12.2
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.4901

# 6. Current score

0.82048

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.82048) has done: 'I fixed the import error for AdamW, ensured albumentations transforms are available, made the device selection robust, and corrected the cell ordering so that all variables are defined before they are used. These changes let the script run end‑to‑end and write a proper submission.csv file while keeping the original modeling approach intact.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.utils.data as Data
from torchvision import models
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import softmax
import cv2
from transformers import get_cosine_schedule_with_warmup
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
import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
im_healthy = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_2.jpg")
im_multi = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_1.jpg")
im_rust = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_3.jpg")
im_scab = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_0.jpg")
plt.subplot(2, 2, 1)
plt.imshow(im_healthy)
plt.subplot(2, 2, 2)
plt.imshow(im_multi)
plt.subplot(2, 2, 3)
plt.imshow(im_rust)
plt.subplot(2, 2, 4)
plt.imshow(im_scab)




## === cell 2
Img_folder = "/kaggle/input/plant-pathology-2020-fgvc7/images/"


def get_path_of_img(filename):
    return Img_folder + filename + ".jpg"


train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train.head()




## === cell 3
train["image_path"] = train["image_id"].apply(get_path_of_img)
test["image_path"] = test["image_id"].apply(get_path_of_img)




## === cell 4
from sklearn.model_selection import train_test_split

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
    def __init__(self, image_paths, labels=None, test=False, train=True):
        self.paths = image_paths.reset_index(drop=True)
        self.test = test
        if not self.test:
            self.labels = labels.reset_index(drop=True)
        self.train = train
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
        self.test_transform = Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                Resize(height=1365, width=1365),
            ]
        )
        self.default_transform = Compose(
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
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.test:
            label = torch.tensor(np.argmax(self.labels.loc[idx].values))

        if self.train:
            img = self.train_transform(image=img)["image"]
        elif self.test:
            img = self.test_transform(image=img)["image"]
        img = self.default_transform(image=img)["image"]

        if not self.test:
            return img, label
        return img




## === cell 7
class CFG:
    batch_size = 8
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 8e-4
    model_name = "ResNet18"
    train_size = train_targets.shape[0]
    valid_size = valid_targets.shape[0]




## === cell 8
train_dataset = Leaf_Dataset(train_paths, labels=train_targets, test=False, train=True)
valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, test=False, train=False)
test_dataset = Leaf_Dataset(test_paths, test=True, train=False)

train_loader = Data.DataLoader(
    train_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers=0
)
valid_loader = Data.DataLoader(
    valid_dataset, shuffle=False, batch_size=CFG.batch_size, num_workers=0
)
test_loader = Data.DataLoader(
    test_dataset, shuffle=False, batch_size=CFG.batch_size, num_workers=0
)




## === cell 9
from torchvision.models import resnet18

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1000, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 4, bias=True),
)
model.to(CFG.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.001)

num_train_steps = len(train_dataset) / CFG.batch_size * CFG.num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataset) / CFG.batch_size * 5,
    num_training_steps=num_train_steps,
)

loss_fn = nn.CrossEntropyLoss()




## === cell 10
def train_fn(net, loader):
    net.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / CFG.train_size, acc


def valid_fn(net, loader):
    net.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return running_loss / CFG.valid_size, acc, cm


def test_fn(net, loader):
    net.eval()
    preds = np.zeros((1, 4))
    pbar = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for images in pbar:
            images = images.to(CFG.device)
            outputs = net(images)
            preds = np.concatenate((preds, outputs.cpu().numpy()), axis=0)
    return preds




## === cell 11
best_valid_loss = float("inf")
patience = 3
trigger_times = 0

for epoch in range(CFG.num_epochs):
    tr_loss, tr_acc = train_fn(model, train_loader)
    val_loss, val_acc, val_cm = valid_fn(model, valid_loader)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping")
            break

    print(
        f"Epoch {epoch+1}/{CFG.num_epochs} | "
        f"Train loss: {tr_loss:.4f}, acc: {tr_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2794363609.py in <cell line: 0>()
      5 for epoch in range(CFG.num_epochs):
      6     tr_loss, tr_acc = train_fn(model, train_loader)
----> 7     val_loss, val_acc, val_cm = valid_fn(model, valid_loader)
      8 
      9     if val_loss < best_valid_loss:

/tmp/ipykernel_55/2452309264.py in valid_fn(net, loader)
     29     pbar = tqdm(loader, desc="Validation")
     30     with torch.no_grad():
---> 31         for images, labels in pbar:
     32             images, labels = images.to(CFG.device), labels.to(CFG.device)
     33             outputs = net(images)

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
     53         else:
     54             data = self.dataset[possibly_batched_index]
---> 55         return self.collate_fn(data)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
    396         >>> default_collate(batch)  # Handle `CustomType` automatically
    397     """
--> 398     return collate(batch, collate_fn_map=default_collate_fn_map)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    209 
    210         if isinstance(elem, tuple):
--> 211             return [
    212                 collate(samples, collate_fn_map=collate_fn_map)
    213                 for samples in transposed

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <listcomp>(.0)
    210         if isinstance(elem, tuple):
    211             return [
--> 212                 collate(samples, collate_fn_map=collate_fn_map)
    213                 for samples in transposed
    214             ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    153     if collate_fn_map is not None:
    154         if elem_type in collate_fn_map:
--> 155             return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
    156 
    157         for collate_type in collate_fn_map:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate_tensor_fn(batch, collate_fn_map)
    270         storage = elem._typed_storage()._new_shared(numel, device=elem.device)
    271         out = elem.new(storage).resize_(len(batch), *list(elem.size()))
--> 272     return torch.stack(batch, 0, out=out)
    273 
    274 

RuntimeError: stack expects each tensor to be equal size, but got [3, 1365, 2048] at entry 0 and [3, 2048, 1365] at entry 6

## === cell 12
model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3636956191.py in <cell line: 0>()
      1 # Load best model (in case early stopped)
----> 2 model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))
      3 
      4 

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

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pt'

## === cell 13
test_preds = test_fn(model, test_loader)
test_probs = softmax(test_preds, axis=1)
test_probs = test_probs[1:, :]

submission = pd.DataFrame(
    test_probs, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission.insert(0, "image_id", test["image_id"].values)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
