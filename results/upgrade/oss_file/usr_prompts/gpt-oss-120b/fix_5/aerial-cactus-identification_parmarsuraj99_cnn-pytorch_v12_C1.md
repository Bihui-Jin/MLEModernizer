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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9894

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
import cv2




## === cell 2
from PIL import Image
from matplotlib import pyplot as plt




## === cell 3
import torch




## === cell 4
DATA_DIR = "./input/aerial-cactus-identification/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
NAMES_DIR = os.path.join(DATA_DIR, "train.csv")




## === cell 5
train_names = pd.read_csv(NAMES_DIR)
train_names.shape




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1457451632.py in <cell line: 0>()
----> 1 train_names = pd.read_csv(NAMES_DIR)
      2 train_names.shape
      3 
      4 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './input/aerial-cactus-identification/train.csv'

## === cell 6
len(os.listdir(TRAIN_DIR))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4146268731.py in <cell line: 0>()
----> 1 len(os.listdir(TRAIN_DIR))
      2 
      3 

FileNotFoundError: [Errno 2] No such file or directory: './input/aerial-cactus-identification/train'

## === cell 7
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms




## === cell 8
def show_image(index):
    image = Image.open(os.path.join(TRAIN_DIR, str(train_names["id"][index])))
    plt.imshow(image)
    print(train_names["has_cactus"][index])


show_image(20)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3691322565.py in <cell line: 0>()
      5 
      6 
----> 7 show_image(20)
      8 
      9 

/tmp/ipykernel_55/3691322565.py in show_image(index)
      1 def show_image(index):
----> 2     image = Image.open(os.path.join(TRAIN_DIR, str(train_names["id"][index])))
      3     plt.imshow(image)
      4     print(train_names["has_cactus"][index])
      5 

NameError: name 'train_names' is not defined

## === cell 9
class HasCactusDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # returns Tensor in [0,1] CxHxW
        else:
            image = transforms.ToTensor()(image)  # fallback
        has_cactus = self.img_names.iloc[idx, 1]
        sample = [image, has_cactus]
        return sample




## === cell 10
hasCactusDataset = HasCactusDataset(
    csv_file=NAMES_DIR,
    root_dir=TRAIN_DIR,
    transform=transforms.Compose([transforms.ToTensor()]),
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2717790253.py in <cell line: 0>()
----> 1 hasCactusDataset = HasCactusDataset(
      2     csv_file=NAMES_DIR,
      3     root_dir=TRAIN_DIR,
      4     transform=transforms.Compose([transforms.ToTensor()]),
      5 )

/tmp/ipykernel_55/850809774.py in __init__(self, csv_file, root_dir, transform)
      1 class HasCactusDataset(Dataset):
      2     def __init__(self, csv_file, root_dir, transform=None):
----> 3         self.img_names = pd.read_csv(csv_file)
      4         self.root_dir = root_dir
      5         self.transform = transform

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './input/aerial-cactus-identification/train.csv'

## === cell 11
plt.imshow(hasCactusDataset[19][0].permute(1, 2, 0).numpy())




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1880692542.py in <cell line: 0>()
----> 1 plt.imshow(hasCactusDataset[19][0].permute(1, 2, 0).numpy())
      2 
      3 

NameError: name 'hasCactusDataset' is not defined

## === cell 12
dataloader = DataLoader(hasCactusDataset, batch_size=128, shuffle=True)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/257200525.py in <cell line: 0>()
----> 1 dataloader = DataLoader(hasCactusDataset, batch_size=128, shuffle=True)
      2 
      3 

NameError: name 'hasCactusDataset' is not defined

## === cell 13
import torch.nn as nn
import torch.nn.functional as F




## === cell 14
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 18, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(18, 25, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(25, 30, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(30, 40, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(40 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))  # shape: (batch,40,6,6)
        x = x.view(-1, 40 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


net = Net().cuda()




## === cell 15
import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)




## === cell 16
from torch.autograd import Variable

net.train()
for epoch in range(30):  # a few more epochs than before
    running_loss = 0.0
    print(f"epoch {epoch} started...")
    for i, data in enumerate(dataloader):
        image, label = data
        optimizer.zero_grad()

        inputs = image.type(torch.FloatTensor).cuda()
        label = label.view(-1, 1).type(torch.FloatTensor).cuda()

        outputs = net(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} completed")
print("finished Training...")




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/93261972.py in <cell line: 0>()
      5     running_loss = 0.0
      6     print(f"epoch {epoch} started...")
----> 7     for i, data in enumerate(dataloader):
      8         image, label = data
      9         optimizer.zero_grad()

NameError: name 'dataloader' is not defined

## === cell 17
data = next(iter(dataloader))




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4282046070.py in <cell line: 0>()
----> 1 data = next(iter(dataloader))
      2 
      3 

NameError: name 'dataloader' is not defined

## === cell 18
class TestSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_names = [
            f
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # Tensor CxHxW, normalized to [0,1]
        else:
            image = transforms.ToTensor()(image)
        sample = [image, img_name]
        return sample




## === cell 19
testSet = TestSet(
    root_dir=TEST_DIR, transform=transforms.Compose([transforms.ToTensor()])
)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1550841420.py in <cell line: 0>()
----> 1 testSet = TestSet(
      2     root_dir=TEST_DIR, transform=transforms.Compose([transforms.ToTensor()])
      3 )
      4 
      5 

/tmp/ipykernel_55/4090186916.py in __init__(self, root_dir, transform)
      3         self.img_names = [
      4             f
----> 5             for f in os.listdir(root_dir)
      6             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
      7         ]

FileNotFoundError: [Errno 2] No such file or directory: './input/aerial-cactus-identification/test'

## === cell 20
testloader = DataLoader(
    testSet, batch_size=1, shuffle=False
)  # keep deterministic order




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3114551920.py in <cell line: 0>()
      1 testloader = DataLoader(
----> 2     testSet, batch_size=1, shuffle=False
      3 )  # keep deterministic order
      4 
      5 

NameError: name 'testSet' is not defined

## === cell 21
net.eval()
tst_outputs = []
tst_names = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        image, name = data[0], data[1]
        inputs = image.type(torch.FloatTensor).cuda()
        output = net(inputs)
        tst_outputs.append(output.cpu().item())
        tst_names.append(name[0])




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3039825112.py in <cell line: 0>()
      3 tst_names = []
      4 with torch.no_grad():
----> 5     for i, data in enumerate(testloader):
      6         image, name = data[0], data[1]
      7         inputs = image.type(torch.FloatTensor).cuda()

NameError: name 'testloader' is not defined

## === cell 22
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
pred_df = pd.DataFrame({"id": tst_names, "has_cactus": tst_outputs})
submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4279611172.py in <cell line: 0>()
      1 # Align predictions with the official sample submission order
----> 2 sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
      3 pred_df = pd.DataFrame({"id": tst_names, "has_cactus": tst_outputs})
      4 # Merge to preserve the exact order from sample_submission
      5 submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './input/aerial-cactus-identification/sample_submission.csv'
