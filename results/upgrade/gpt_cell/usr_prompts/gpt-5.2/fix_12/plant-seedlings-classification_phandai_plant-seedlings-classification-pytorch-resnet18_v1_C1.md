# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.13

# 2. Installed packages

geopandas==0.14.4
kaggle==1.7.4.5
kaggle-environments==1.18.0
kagglehub==0.3.13
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 4. Code solution

## === cell 0
import os
import glob
from pathlib import Path
import random
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torchinfo


## === cell 1
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

COMP_NAME = 'plant-seedlings-classification'
if COMP_NAME is None:
    raise NameError('COMP_NAME has not been initialized')

DATA_PATH = Path('../input/' + COMP_NAME) if IS_KAGGLE else Path('./data')

RANDOM_SEED = 42
BATCH_SIZE = 32

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


## === cell 2
print('kaggle:', 'Y' if IS_KAGGLE else 'N')
print('torch version:', torch.__version__)
print('device:', DEVICE)
print(torch.cuda.device_count(), 'GPU(s) available')


## === cell 3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


## === cell 4
local_comp_path = Path("./data") / COMP_NAME
if local_comp_path.exists():
    DATA_PATH = local_comp_path

if (
    (DATA_PATH / COMP_NAME).exists()
    and (DATA_PATH / COMP_NAME / "train").exists()
    and (DATA_PATH / COMP_NAME / "test").exists()
):
    DATA_PATH = DATA_PATH / COMP_NAME

nested_comp_path = DATA_PATH / COMP_NAME
if (nested_comp_path / "train").exists() and (nested_comp_path / "test").exists():
    DATA_PATH = nested_comp_path

candidate_paths = [
    DATA_PATH,  # current
    Path("./data") / COMP_NAME,  # explicit local competition folder
    Path("./data") / COMP_NAME / COMP_NAME,  # sometimes nested under same name
    Path("./kaggle") / "input",  # environment-provided root (relative)
    Path("./kaggle") / "input" / COMP_NAME,  # environment-provided location (relative)
    Path("./kaggle") / "input" / COMP_NAME / COMP_NAME,  # nested under input (relative)
    Path("./kaggle")
    / "input"
    / COMP_NAME
    / COMP_NAME
    / COMP_NAME,  # deeper nesting seen in exports (relative)
    Path("./kaggle") / "data",  # environment-provided root (relative)
    Path("./kaggle") / "data" / COMP_NAME,  # environment-provided location (relative)
    Path("./kaggle")
    / "data"
    / COMP_NAME
    / COMP_NAME,  # common double-nested layout (relative)
    Path("./kaggle")
    / "data"
    / COMP_NAME
    / COMP_NAME
    / COMP_NAME,  # deeper nesting seen in exports (relative)
    Path("./kaggle")
    / "data"
    / COMP_NAME
    / COMP_NAME
    / COMP_NAME
    / COMP_NAME,  # even deeper nesting (rare, but safe) (relative)
    Path("/kaggle") / "input",
    Path("/kaggle") / "input" / COMP_NAME,
    Path("/kaggle") / "input" / COMP_NAME / COMP_NAME,
    Path("/kaggle") / "data",
    Path("/kaggle") / "data" / COMP_NAME,
    Path("/kaggle") / "data" / COMP_NAME / COMP_NAME,
    Path("/kaggle") / "working",
    Path("/kaggle") / "working" / COMP_NAME,
    Path("/kaggle") / "working" / COMP_NAME / COMP_NAME,
]
for p in candidate_paths:
    if (p / "train").exists() and (p / "test").exists():
        DATA_PATH = p
        break
    if (p / COMP_NAME / "train").exists() and (p / COMP_NAME / "test").exists():
        DATA_PATH = p / COMP_NAME
        break

if not (DATA_PATH / "train").exists() or not (DATA_PATH / "test").exists():
    if IS_KAGGLE:
        import zipfile, kaggle

        kaggle.api.competition_download_cli(COMP_NAME)
        zipfile.ZipFile(f"{COMP_NAME}.zip").extractall(DATA_PATH)
    else:
        raise FileNotFoundError(
            f"Could not locate dataset folders under DATA_PATH={DATA_PATH!s}. "
            f"Expected '{DATA_PATH / 'train'}' and '{DATA_PATH / 'test'}' to exist."
        )


## === cell 5
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std),
])

all_ds = datasets.ImageFolder(root=DATA_PATH/'train', transform=transform)


## === cell 6
all_samples = len(all_ds)

print(all_samples, 'samples')
print(len(all_ds.classes), 'labels')


## === cell 7
label_counts = []

for d in glob.glob(os.path.join(DATA_PATH/'train', '*')):
    label = os.path.basename(d)
    count = len(glob.glob(os.path.join(d, '*')))
    label_counts.append({'label': label, 'count': count})

label_counts_df = pd.DataFrame(label_counts)
print(label_counts_df)


## === cell 8
plt.figure(figsize=(8, 5))
plt.barh(label_counts_df['label'], label_counts_df['count'])
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Label Counts')
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()


## === cell 9
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

labels = all_ds.classes

for i in range(1, cols * rows + 1):
    sample = all_ds[random.randint(0, len(all_ds)-1)]
    label = sample[1]
    
    img = sample[0].permute(1, 2, 0) # (3, 224, 224) -> (224, 224, 3)
    img = transform_std * np.array(img) + transform_mean # undo normalization
    
    figure.add_subplot(rows, cols, i)
    plt.title(labels[label])
    plt.axis('off')
    plt.imshow(img)

plt.show()


## === cell 10
all_len = len(all_ds)
valid_len = 1000 if all_len >= 1000 else max(1, int(0.2 * all_len))
train_len = all_len - valid_len

train_ds, valid_ds = torch.utils.data.random_split(all_ds, [train_len, valid_len])

print("train:", len(train_ds), "samples")
print("valid:", len(valid_ds), "samples")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


## === cell 11
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


## === cell 12
model


## === cell 13
input_features = model.fc.in_features
model.fc = nn.Linear(input_features, len(all_ds.classes), device=DEVICE)
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)

print(model)
torchinfo.summary(model, (BATCH_SIZE, 3, 224, 224), col_names=('input_size', 'output_size', 'num_params', 'kernel_size'), verbose=0)


## === cell 14
def train_step(dataloader, model, loss_fn, optimizer, print_every=100):
    losses = []
    model.train()

    for batch, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch % print_every == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            print(f'  Training: Loss = {loss:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]')
    return losses


## === cell 15
def valid_step(dataloader, model, loss_fn):
    model.eval()
    loss, correct = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            loss += loss_fn(outputs, targets).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets.data).sum().item()

    loss /= BATCH_SIZE
    correct /= len(dataloader.dataset)
    print(f'  Validation: Accuracy={(100 * correct):>0.1f}%, Average_Loss={loss:>8f}\n')


## === cell 16
epochs = 10
losses = []

for epoch in range(epochs):
    print(f'Epoch [{epoch+1:>2d}/{epochs}]\n-------------------------------')
    epoch_losses = train_step(train_loader, model, loss_fn, optimizer, print_every=25)
    valid_step(valid_loader, model, loss_fn)
    losses.extend(epoch_losses)


## === cell 17
plt.figure(figsize=(12, 4))

plt.plot(losses)
plt.title('Cross Entropy Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()


## === cell 18
class TestDataset(Dataset):
    def __init__(self, test_path, transform=None):
        self.test_path = test_path
        self.transform = transform

    def __len__(self):
        return len(glob.glob(f'{self.test_path}/*.png'))

    def __getitem__(self, index):
        img_path = os.path.join(self.test_path, os.listdir(self.test_path)[index])
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img


## === cell 19
test_path = DATA_PATH/'test'

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std)
])

test_ds = TestDataset(test_path, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print('Test:', len(test_ds), 'samples')


## === cell 20
labels = []

model.eval()

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        labels.extend(preds.cpu().numpy().tolist())

species = [all_ds.classes[label] for label in labels]

submission = pd.DataFrame({'file': os.listdir(test_path), 'species': species})
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3306943676.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m     [0;32mfor[0m [0minputs[0m [0;32min[0m [0mtest_loader[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m         [0minputs[0m [0;34m=[0m [0minputs[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mDEVICE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m         [0moutputs[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0minputs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m    762[0m     [0;32mdef[0m [0m_next_data[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    763[0m         [0mindex[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_index[0m[0;34m([0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 764[0;31m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    766[0m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36mfetch[0;34m(self, possibly_batched_index)[0m
[1;32m     50[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0m__getitems__[0m[0;34m([0m[0mpossibly_batched_index[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m                 [0mdata[0m [0;34m=[0m [0;34m[[0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0midx[0m[0;34m][0m [0;32mfor[0m [0midx[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     53[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     50[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0m__getitems__[0m[0;34m([0m[0mpossibly_batched_index[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m                 [0mdata[0m [0;34m=[0m [0;34m[[0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0midx[0m[0;34m][0m [0;32mfor[0m [0midx[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     53[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3945817129.py[0m in [0;36m__getitem__[0;34m(self, index)[0m
[1;32m      9[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mindex[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m         [0mimg_path[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtest_path[0m[0;34m,[0m [0mos[0m[0;34m.[0m[0mlistdir[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtest_path[0m[0;34m)[0m[0;34m[[0m[0mindex[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m         [0mimg[0m [0;34m=[0m [0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mimg_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m [0;34m[0m[0m
[1;32m     13[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mtransform[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/PIL/Image.py[0m in [0;36mopen[0;34m(fp, mode, formats)[0m
[1;32m   3511[0m     [0;32mif[0m [0mis_path[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3512[0m         [0mfilename[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mfspath[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3513[0;31m         [0mfp[0m [0;34m=[0m [0mbuiltins[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3514[0m         [0mexclusive_fp[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3515[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIsADirectoryError[0m: [Errno 21] Is a directory: '/kaggle/input/test/test'

## === cell 21
%cd /kaggle
