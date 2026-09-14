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

0.96082

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize

import glob
import random
import zipfile
from PIL import Image
import matplotlib.pyplot as plt


## === cell 2
SEED = 42

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


## === cell 3
glob.glob("/kaggle/input/paddy-disease-classification/*")


## === cell 4
set_seed(SEED)

train_val_images = glob.glob("/kaggle/input/paddy-disease-classification/train_images/*/*")
test_images = glob.glob("/kaggle/input/paddy-disease-classification/test_images/*")


def train_val_split(images_list, train_size):
    n = int(len(images_list) * train_size)
    train_list = images_list[:n]
    val_list = images_list[n:]
    return train_list, val_list

random.shuffle(train_val_images)
train_images, val_images = train_val_split(train_val_images, train_size=0.9)

print(len(train_images), len(val_images), len(test_images))

train_images[:2], val_images[:2], test_images[:2]


## === cell 5
transform = Compose([
    CenterCrop(446),
    Resize((256, 256)),
    ToTensor(),
    Normalize([0.4965, 0.5858, 0.2238], [0.2222, 0.2218, 0.1968])
])


## === cell 6
target_names = [st.split("/")[-1] for st in glob.glob("/kaggle/input/paddy-disease-classification/train_images/*")]
target_names.sort()
target_names


## === cell 7
target_2_int = {}
counter = 0
for _ in target_names:
    target_2_int[_] = counter
    counter += 1

target_2_int


## === cell 8
int_2_target = {i:t for t, i in target_2_int.items()}

int_2_target


## === cell 9
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = img_path.split("/")[-2]
            label = target_2_int[label]
        else:
            label = img_path.split("/")[-1].split(".")[0]
        
        label = int(label)
        return image, label


## === cell 11
training_data = ImageDataset(train_images, transform, train=True)
val_data = ImageDataset(val_images, transform, train=True)

test_data = ImageDataset(test_images, transform, train=False)


## === cell 12
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


## === cell 13
for x, y in train_dataloader:
    print(x.shape)
    print(y)
    break


## === cell 14
for x, y in val_dataloader:
    print(x.shape)
    print(y)
    break


## === cell 15
for x, y in test_dataloader:
    print(x.shape)
    print(y)
    break


## === cell 16
fig = plt.figure(figsize=(6, 6))
img = make_grid(next(iter(train_dataloader))[0], nrow=4)
img = np.transpose(img, (1, 2, 0))
plt.imshow(img)
plt.axis("off")
plt.show()


## === cell 17
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()else "cpu"
print(f"Using {device} device")


## === cell 19
set_seed(SEED)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = nn.Linear(512, 10)
model.to(device);


## === cell 20
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


## === cell 21
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
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


## === cell 22
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


## === cell 23
epochs = 3

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done!")


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3180107300.py in <cell line: 0>()
      3 for t in range(epochs):
      4     print(f"Epoch {t+1}\n-------------------------------")
----> 5     train(train_dataloader, model, loss_fn, optimizer)
      6     test(val_dataloader, model, loss_fn)
      7 print("Done!")

/tmp/ipykernel_11/1850887561.py in train(dataloader, model, loss_fn, optimizer)
     11         # Backpropagation
     12         optimizer.zero_grad()
---> 13         loss.backward()
     14         optimizer.step()
     15 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    338 
    339     grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
--> 340     grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    341     if retain_graph is None:
    342         retain_graph = create_graph

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in _make_grads(outputs, grads, is_grads_batched)
    218                     assert isinstance(out, torch.Tensor)
    219                     new_grads.append(
--> 220                         torch.ones_like(out, memory_format=torch.preserve_format)
    221                     )
    222             else:

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## === cell 26
test_label = np.zeros((len(test_dataloader.dataset,)), dtype=object)
test_id = np.zeros((len(test_dataloader.dataset,)), dtype=object)


for i, (X, y) in enumerate(test_dataloader):
    X = X.to(device)
    model.eval()
    pred = model(X)
    test_label[i] = int_2_target[torch.argmax(pred, dim=1).item()]
    test_id[i] = str(y.item()) + ".jpg"


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2379233628.py in <cell line: 0>()
      4 
      5 for i, (X, y) in enumerate(test_dataloader):
----> 6     X = X.to(device)
      7     model.eval()
      8     pred = model(X)

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## === cell 27
sub_df = pd.DataFrame({"image_id": test_id, "label": test_label})


## === cell 28
sub_df.head()


## === cell 29
sub_df.to_csv("submission.csv", index=False)


## --- ERROR in outputing the csv:
Invalid submission: Expected 2602 rows, but got 2603
