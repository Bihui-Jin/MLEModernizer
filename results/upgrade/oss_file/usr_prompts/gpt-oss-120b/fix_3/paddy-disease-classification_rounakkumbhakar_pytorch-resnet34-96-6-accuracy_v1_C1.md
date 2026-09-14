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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.80645

# 6. Current score

0.16833

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.16833) has done: 'Implemented fixes to eliminate runtime CUDA errors and ensure a full submission is generated.  
- Forced computation onto CPU to sidestep cuDNN stream mismatches.  
- Adjusted the inference loop to correctly extract filenames from the dataloader batch (now handling the list returned for each image).  
- Added a brief safety comment and retained the original workflow and architecture.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os




## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

import glob
import random
import zipfile
from PIL import Image
import matplotlib.pyplot as plt




## === cell 2
glob.glob("/kaggle/input/paddy-disease-classification/*")




## === cell 3
train_val_images = glob.glob(
    "/kaggle/input/paddy-disease-classification/train_images/*/*"
)
test_images = glob.glob("/kaggle/input/paddy-disease-classification/test_images/*.jpg")




## === cell 4
def train_val_split(images_list, train_size):
    n = int(len(images_list) * train_size)
    train_list = images_list[:n]
    val_list = images_list[n:]
    return train_list, val_list


train_images, val_images = train_val_split(train_val_images, train_size=0.9)

len(train_images), len(val_images)




## === cell 5
len(train_images), len(val_images), len(test_images)




## === cell 6
random.shuffle(train_images)
random.shuffle(val_images)
random.shuffle(test_images)




## === cell 7
train_images[:2]




## === cell 8
val_images[:2]




## === cell 9
test_images[:2]




## === cell 10
transform = Compose([Resize((128, 128)), ToTensor()])




## === cell 11
target_names = [
    st.split("/")[-1]
    for st in glob.glob("/kaggle/input/paddy-disease-classification/train_images/*")
]
target_names.sort()
target_names




## === cell 12
target_2_int = {}
counter = 0
for _ in target_names:
    target_2_int[_] = counter
    counter += 1

target_2_int




## === cell 13
int_2_target = {i: t for t, i in target_2_int.items()}

int_2_target




## === cell 14
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.train:
            label_str = img_path.split("/")[-2]
            label = target_2_int[label_str]
            return image, label
        else:
            filename = os.path.basename(img_path)
            return image, filename




## === cell 15
training_data = ImageDataset(train_images, transform, train=True)
val_data = ImageDataset(val_images, transform, train=True)

test_data = ImageDataset(test_images, transform, train=False)




## === cell 16
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)




## === cell 17
for x, y in train_dataloader:
    print(x.shape)
    print(y)
    break




## === cell 18
for x, y in test_dataloader:
    print(x.shape)
    print(y)
    break




## === cell 19
fig = plt.figure(figsize=(6, 6))
columns = 4
rows = 4
images = next(iter(train_dataloader))[0]
for i in range(1, columns * rows + 1):
    img = np.transpose(images[i - 1], (1, 2, 0))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis("off")
plt.show()




## === cell 20
device = torch.device("cpu")
print(f"Using {device} device")




## === cell 21
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 disease classes
model = model.to(device)
model




## === cell 22
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)




## === cell 23
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")




## === cell 24
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )




## === cell 25
epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done!")




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/1050152332.py in <cell line: 0>()
      2 for t in range(epochs):
      3     print(f"Epoch {t+1}\n-------------------------------")
----> 4     train(train_dataloader, model, loss_fn, optimizer)
      5     test(val_dataloader, model, loss_fn)
      6 print("Done!")

/tmp/ipykernel_55/1073786887.py in train(dataloader, model, loss_fn, optimizer)
      6 
      7         pred = model(X)
----> 8         loss = loss_fn(pred, y)
      9 
     10         optimizer.zero_grad()

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/loss.py in forward(self, input, target)
   1293 
   1294     def forward(self, input: Tensor, target: Tensor) -> Tensor:
-> 1295         return F.cross_entropy(
   1296             input,
   1297             target,

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
   3492     if size_average is not None or reduce is not None:
   3493         reduction = _Reduction.legacy_get_string(size_average, reduce)
-> 3494     return torch._C._nn.cross_entropy_loss(
   3495         input,
   3496         target,

IndexError: Target 10 is out of bounds.

## === cell 26
model.eval()
pred_labels = []
pred_ids = []

with torch.no_grad():
    for X, img_name in test_dataloader:
        X = X.to(device)
        outputs = model(X)
        pred_idx = outputs.argmax(dim=1).item()
        pred_labels.append(int_2_target[pred_idx])
        pred_ids.append(img_name[0])

print("Number of predictions:", len(pred_labels))




## === cell 27
sub_df = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})
print("Submission shape:", sub_df.shape)
assert sub_df.shape[0] == 2602, f"Expected 2602 rows, got {sub_df.shape[0]}"




## === cell 28
sub_df.head()




## === cell 29
sub_df.to_csv("submission.csv", index=False)
print("submission.csv written with", sub_df.shape[0], "rows.")
