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

3.12

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
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

0.9886

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import zipfile
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import random
import shutil
from torchvision.transforms import ToTensor


## === cell 2
train_file_path = '/kaggle/input/aerial-cactus-identification/train.zip'
image_dir = '/kaggle/working/'

with zipfile.ZipFile(train_file_path, 'r') as zip_ref:
    zip_ref.extractall(image_dir)
    
train_list = os.listdir('/kaggle/working/train')


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4268454130.py in <cell line: 0>()
      5     zip_ref.extractall(image_dir)
      6 
----> 7 train_list = os.listdir('/kaggle/working/train')

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train'

## === cell 3
len(train_list)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1982950360.py in <cell line: 0>()
----> 1 len(train_list)

NameError: name 'train_list' is not defined

## === cell 4
for i, file_name in enumerate(train_list[:10]):
    print(f"{i+1}: {file_name}")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3693885805.py in <cell line: 0>()
----> 1 for i, file_name in enumerate(train_list[:10]):
      2     print(f"{i+1}: {file_name}")

NameError: name 'train_list' is not defined

## === cell 5
test_file_path = '/kaggle/input/aerial-cactus-identification/test.zip'

with zipfile.ZipFile(test_file_path, 'r') as zip_ref:
    zip_ref.extractall(image_dir)
    
test_list = os.listdir('/kaggle/working/test')


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2193712832.py in <cell line: 0>()
      4     zip_ref.extractall(image_dir)
      5 
----> 6 test_list = os.listdir('/kaggle/working/test')

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test'

## === cell 6
len(test_list)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2057636824.py in <cell line: 0>()
----> 1 len(test_list)

NameError: name 'test_list' is not defined

## === cell 7
for i, file_name in enumerate(test_list[:10]):
    print(f"{i+1}: {file_name}")


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1150937817.py in <cell line: 0>()
----> 1 for i, file_name in enumerate(test_list[:10]):
      2     print(f"{i+1}: {file_name}")

NameError: name 'test_list' is not defined

## === cell 8
train_image_file_path = get_image_files('/kaggle/working/train')
train_image_file_path[0]


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/4279022908.py in <cell line: 0>()
      1 train_image_file_path = get_image_files('/kaggle/working/train')
----> 2 train_image_file_path[0]

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, idx)
    118     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    119     def __getitem__(self, idx):
--> 120         if isinstance(idx,int) and not hasattr(self.items,'iloc'): return self.items[idx]
    121         return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    122     def copy(self): return self._new(self.items.copy())

IndexError: list index out of range

## === cell 9
im = Image.open(train_image_file_path[0])
im


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1453218573.py in <cell line: 0>()
----> 1 im = Image.open(train_image_file_path[0])
      2 im

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, idx)
    118     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    119     def __getitem__(self, idx):
--> 120         if isinstance(idx,int) and not hasattr(self.items,'iloc'): return self.items[idx]
    121         return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    122     def copy(self): return self._new(self.items.copy())

IndexError: list index out of range

## === cell 10
test_image_file_path = get_image_files('/kaggle/working/test')
test_image_file_path[0]


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1781057987.py in <cell line: 0>()
      1 test_image_file_path = get_image_files('/kaggle/working/test')
----> 2 test_image_file_path[0]

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, idx)
    118     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    119     def __getitem__(self, idx):
--> 120         if isinstance(idx,int) and not hasattr(self.items,'iloc'): return self.items[idx]
    121         return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    122     def copy(self): return self._new(self.items.copy())

IndexError: list index out of range

## === cell 11
im2 = Image.open(test_image_file_path[0])
im2


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2809083192.py in <cell line: 0>()
----> 1 im2 = Image.open(test_image_file_path[0])
      2 im2

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, idx)
    118     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    119     def __getitem__(self, idx):
--> 120         if isinstance(idx,int) and not hasattr(self.items,'iloc'): return self.items[idx]
    121         return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    122     def copy(self): return self._new(self.items.copy())

IndexError: list index out of range

## === cell 12
train_csv = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
test_csv = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')


## === cell 13
train_csv.head()


## === cell 14
test_csv.head()


## === cell 15
train_csv[train_csv['has_cactus']==1]


## === cell 16
train_csv[train_csv['has_cactus']==0]


## === cell 18
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, df, transform=None):  
        self.path = path
        self.df = df
        self.transform = transform      
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img_id = self.df.iloc[i, 0]
        
        img = Image.open(self.path + img_id).convert('RGB')
        label = self.df.iloc[i, 1]
        
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


## === cell 22
from sklearn.model_selection import train_test_split

train, valid = train_test_split(train_csv, test_size=0.1, stratify = train_csv['has_cactus'])


## === cell 23
train.iloc[0, 0]


## === cell 24
train.iloc[0, 1]


## === cell 25
len(train), len(valid)


## === cell 26
valid['has_cactus'].value_counts()


## === cell 27
print('/kaggle/working/train/' + train.iloc[0, 0])


## === cell 32
train_ds, valid_ds = CustomDataset(path='/kaggle/working/train/', df=train, transform=ToTensor()), CustomDataset(path='/kaggle/working/train/', df=valid, transform=ToTensor())


## === cell 33
len(train_ds)


## === cell 34
x, y = train_ds[0]


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3197885.py in <cell line: 0>()
      1 #데이터셋 확인하기
----> 2 x, y = train_ds[0]

/tmp/ipykernel_11/1385111463.py in __getitem__(self, i)
     14         img_id = self.df.iloc[i, 0]
     15 
---> 16         img = Image.open(self.path + img_id).convert('RGB')
     17         label = self.df.iloc[i, 1]
     18 

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/6c6cff978b8a3ffb8b6e6a243dc4c5ce.jpg'

## === cell 35
x.shape, y


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2745103210.py in <cell line: 0>()
----> 1 x.shape, y

NameError: name 'x' is not defined

## === cell 36
test_ds = CustomDataset('/kaggle/working/test/', test_csv, transform=ToTensor())


## === cell 37
z, k = test_ds[0]
z.shape, k


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4209862767.py in <cell line: 0>()
----> 1 z, k = test_ds[0]
      2 z.shape, k

/tmp/ipykernel_11/1385111463.py in __getitem__(self, i)
     14         img_id = self.df.iloc[i, 0]
     15 
---> 16         img = Image.open(self.path + img_id).convert('RGB')
     17         label = self.df.iloc[i, 1]
     18 

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test/09034a34de0e2015a8a28dfe18f423f6.jpg'

## === cell 38
def collate(idxs, ds):
    xb, yb = zip(*[ds[i] for i in idxs])
    return torch.stack(xb), torch.tensor([y for y in yb], dtype=torch.int64)


## === cell 39
x, y = collate([1, 2], train_ds)
x.shape, y


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2763842159.py in <cell line: 0>()
----> 1 x, y = collate([1, 2], train_ds)
      2 x.shape, y

/tmp/ipykernel_11/96376896.py in collate(idxs, ds)
      1 def collate(idxs, ds):
----> 2     xb, yb = zip(*[ds[i] for i in idxs])
      3     return torch.stack(xb), torch.tensor([y for y in yb], dtype=torch.int64)

/tmp/ipykernel_11/96376896.py in <listcomp>(.0)
      1 def collate(idxs, ds):
----> 2     xb, yb = zip(*[ds[i] for i in idxs])
      3     return torch.stack(xb), torch.tensor([y for y in yb], dtype=torch.int64)

/tmp/ipykernel_11/1385111463.py in __getitem__(self, i)
     14         img_id = self.df.iloc[i, 0]
     15 
---> 16         img = Image.open(self.path + img_id).convert('RGB')
     17         label = self.df.iloc[i, 1]
     18 

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/c01b3421de1cc49f448f1cbdce9cd9e4.jpg'

## === cell 40
class DataLoader:
    def __init__(self, ds, bs=64, shuffle=False, n_workers=1):
        self.ds, self.bs, self.shuffle, self.n_workers = ds, bs, shuffle, n_workers
        
    def __len__(self): return (len(self.ds)-1)//self.bs+1
    
    def __iter__(self):
        idxs = L.range(self.ds)
        if self.shuffle: idxs = idxs.shuffle()
        chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
        with ProcessPoolExecutor(self.n_workers) as ex:
            yield from ex.map(collate, chunks, ds=self.ds)


## === cell 41
n_workers = min(16, defaults.cpus)
train_dl = DataLoader(train_ds, bs=64, shuffle=True, n_workers = n_workers)
valid_dl = DataLoader(valid_ds, bs=64, shuffle=False, n_workers = n_workers)

xb, yb = first(train_dl)


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastcore/parallel.py", line 63, in _call
    return g(item)
           ^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in collate
    xb, yb = zip(*[ds[i] for i in idxs])
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in <listcomp>
    xb, yb = zip(*[ds[i] for i in idxs])
                   ~~^^^
  File "/tmp/ipykernel_11/1385111463.py", line 16, in __getitem__
    img = Image.open(self.path + img_id).convert('RGB')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/9f998c6dfc95bea197588aa4377ebeee.jpg'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3883325633.py in <cell line: 0>()
      3 valid_dl = DataLoader(valid_ds, bs=64, shuffle=False, n_workers = n_workers)
      4 
----> 5 xb, yb = first(train_dl)

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in first(x, f, negate, **kwargs)
    740     x = iter(x)
    741     if f: x = filter_ex(x, f=f, negate=negate, gen=True, **kwargs)
--> 742     return next(x, None)
    743 
    744 # %% ../nbs/01_basics.ipynb

/tmp/ipykernel_11/537660781.py in __iter__(self)
     10         chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
     11         with ProcessPoolExecutor(self.n_workers) as ex:
---> 12             yield from ex.map(collate, chunks, ds=self.ds)

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/9f998c6dfc95bea197588aa4377ebeee.jpg'

## === cell 42
test_dl = DataLoader(test_ds, bs=64, shuffle=False, n_workers=n_workers)


## === cell 43
zb, kb = first(test_dl)
zb, kb


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastcore/parallel.py", line 63, in _call
    return g(item)
           ^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in collate
    xb, yb = zip(*[ds[i] for i in idxs])
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in <listcomp>
    xb, yb = zip(*[ds[i] for i in idxs])
                   ~~^^^
  File "/tmp/ipykernel_11/1385111463.py", line 16, in __getitem__
    img = Image.open(self.path + img_id).convert('RGB')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test/09034a34de0e2015a8a28dfe18f423f6.jpg'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3929236896.py in <cell line: 0>()
----> 1 zb, kb = first(test_dl)
      2 zb, kb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in first(x, f, negate, **kwargs)
    740     x = iter(x)
    741     if f: x = filter_ex(x, f=f, negate=negate, gen=True, **kwargs)
--> 742     return next(x, None)
    743 
    744 # %% ../nbs/01_basics.ipynb

/tmp/ipykernel_11/537660781.py in __iter__(self)
     10         chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
     11         with ProcessPoolExecutor(self.n_workers) as ex:
---> 12             yield from ex.map(collate, chunks, ds=self.ds)

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test/09034a34de0e2015a8a28dfe18f423f6.jpg'

## === cell 44
zb.shape, kb.shape


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2569363171.py in <cell line: 0>()
----> 1 zb.shape, kb.shape

NameError: name 'zb' is not defined

## === cell 45
xb, yb


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2865613517.py in <cell line: 0>()
----> 1 xb, yb

NameError: name 'xb' is not defined

## === cell 46
xb.shape, yb.shape


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1800805511.py in <cell line: 0>()
----> 1 xb.shape, yb.shape

NameError: name 'xb' is not defined

## === cell 47
import torch.nn as nn #신경망 모듈
import torch.nn.functional as F #신경망 모듈에서 자주 사용되는 함수 


## === cell 48
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=2),
                                   nn.ReLU(), 
                                   nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=32*4*4, out_features=2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(-1, 32*4*4) #평탄화
        x = self.fc(x)
        
        return x


## === cell 49
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


## === cell 50
device


## === cell 51
model = Model().to(device)


## === cell 52
loss_fn = nn.CrossEntropyLoss()


## === cell 53
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)


## === cell 54
len(train_dl)


## === cell 55
epochs = 9

for epoch in range(epochs):
    epoch_loss = 0
    
    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        pred = model(images)
        
        loss = loss_fn(pred, labels)
        
        epoch_loss += loss.item() #역전파 수행
        loss.backward()
        
        optimizer.step()
        
    print(f'에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(train_dl):.4f}') 


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastcore/parallel.py", line 63, in _call
    return g(item)
           ^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in collate
    xb, yb = zip(*[ds[i] for i in idxs])
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in <listcomp>
    xb, yb = zip(*[ds[i] for i in idxs])
                   ~~^^^
  File "/tmp/ipykernel_11/1385111463.py", line 16, in __getitem__
    img = Image.open(self.path + img_id).convert('RGB')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/66c9e1526646d7b4a5d7135d59cc339e.jpg'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1775301148.py in <cell line: 0>()
      4     epoch_loss = 0
      5 
----> 6     for images, labels in train_dl:
      7         images = images.to(device)
      8         labels = labels.to(device)

/tmp/ipykernel_11/537660781.py in __iter__(self)
     10         chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
     11         with ProcessPoolExecutor(self.n_workers) as ex:
---> 12             yield from ex.map(collate, chunks, ds=self.ds)

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/66c9e1526646d7b4a5d7135d59cc339e.jpg'

## === cell 56
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []


## === cell 57
model.eval()

with torch.no_grad(): #성능 검증 시에는 기울기 계산을 비활성화
    for images, labels in valid_dl:
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        preds = torch.softmax(output.cpu(), dim=1)[:, 1] #예측 확률
        true = labels.cpu() #실젯값
        
        preds_list.extend(preds)
        true_list.extend(true)

        
print(f'검증 데이터 ROC AUC: {roc_auc_score(true_list, preds_list):.4f}')


## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastcore/parallel.py", line 63, in _call
    return g(item)
           ^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in collate
    xb, yb = zip(*[ds[i] for i in idxs])
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in <listcomp>
    xb, yb = zip(*[ds[i] for i in idxs])
                   ~~^^^
  File "/tmp/ipykernel_11/1385111463.py", line 16, in __getitem__
    img = Image.open(self.path + img_id).convert('RGB')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/86420e3a74cc7d1e60a3673068267c3c.jpg'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1189692305.py in <cell line: 0>()
      3 
      4 with torch.no_grad(): #성능 검증 시에는 기울기 계산을 비활성화
----> 5     for images, labels in valid_dl:
      6         images = images.to(device)
      7         labels = labels.to(device)

/tmp/ipykernel_11/537660781.py in __iter__(self)
     10         chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
     11         with ProcessPoolExecutor(self.n_workers) as ex:
---> 12             yield from ex.map(collate, chunks, ds=self.ds)

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/86420e3a74cc7d1e60a3673068267c3c.jpg'

## === cell 58
model.eval()
preds = []

with torch.no_grad():
    for images, _ in test_dl:
        images = images.to(device)
        
        outputs = model(images)
        
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        
        preds.extend(preds_part)


## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastcore/parallel.py", line 63, in _call
    return g(item)
           ^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in collate
    xb, yb = zip(*[ds[i] for i in idxs])
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/96376896.py", line 2, in <listcomp>
    xb, yb = zip(*[ds[i] for i in idxs])
                   ~~^^^
  File "/tmp/ipykernel_11/1385111463.py", line 16, in __getitem__
    img = Image.open(self.path + img_id).convert('RGB')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test/09034a34de0e2015a8a28dfe18f423f6.jpg'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2904155910.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for images, _ in test_dl:
      6         images = images.to(device)
      7 

/tmp/ipykernel_11/537660781.py in __iter__(self)
     10         chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
     11         with ProcessPoolExecutor(self.n_workers) as ex:
---> 12             yield from ex.map(collate, chunks, ds=self.ds)

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test/09034a34de0e2015a8a28dfe18f423f6.jpg'

## === cell 59
test_csv['has_cactus'] = preds
test_csv.to_csv('submission.csv', index=False)


## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1541559400.py in <cell line: 0>()
----> 1 test_csv['has_cactus'] = preds
      2 test_csv.to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (3325)
