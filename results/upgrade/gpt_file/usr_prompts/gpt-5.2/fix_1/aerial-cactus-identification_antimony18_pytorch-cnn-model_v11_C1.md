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

2.7

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.9903

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt


## === cell 1
class CactusImageDataset(Dataset):
    "Aerial Cactus Classification Dataset"
    
    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.cactus_annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.cactus_annotations.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.cactus_annotations.iloc[idx, 0])
        image = io.imread(img_name)
        has_cactus = self.cactus_annotations.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        sample = {"image":image, 'label':has_cactus}
        return sample
    
        


## === cell 2
image_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
}
    


## === cell 3
dataset = CactusImageDataset('../input/aerial-cactus-identification/train.csv','../input/aerial-cactus-identification/train/train/', image_transforms['train'])


## === cell 4
%matplotlib inline
plt.figure(dpi=128, figsize=(3,3))
plt.title('Distribution of Cacti Images')
plt.xticks([0,1])
plt.xlabel('Has Cactus')
plt.ylabel('Number of Images')
dataset.cactus_annotations['has_cactus'].hist(bins=2)


## === cell 5
train_size = int((0.80 * dataset.__len__()))
dev_size = dataset.__len__() - train_size
train_set, dev_set = random_split(dataset, [train_size, dev_size])


## === cell 6
BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)


## === cell 7
class CactusIdentifier(nn.Module):
    """
        Aerial Cactus Identifier Model
    """
    def __init__(self):
        super(CactusIdentifier, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding = 1)
        self.conv2 = nn.Conv2d(16,32,3, padding = 1)
        self.conv3 = nn.Conv2d(32,64,3, padding = 1)
        self.conv4 = nn.Conv2d(64,128,3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128*2*2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
    
    def forward(self, inp_image):
        out = self.pool(self.act(self.conv1(inp_image)))
        out = self.pool(self.act(self.conv2(out)))
        out = self.pool(self.act(self.conv3(out)))
        out = self.pool(self.act(self.conv4(out)))
        
        out = out.view(-1, 128*2*2)
        out = self.drop(out)
        out = self.act(self.fc1(out))
        out = self.drop(out)
        out = self.act(self.fc2(out))
        out = self.drop(out)
        out = self.fc3(out)
        
        return out


## === cell 8
class TrainingModule():
    """
    Training Module to train the model
    """
    
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True
            self.model = model.cuda()
    
    def train_epoch(self, epoch, train_iterator):
        total_loss = 0
        stats_string = "Epoch : {:3d} | Iteration : {:4d} | Loss : {:4.4f}"
        for i, data in enumerate(train_iterator):
            inputs = data['image']
            labels = data['label'].long()
            
            if self.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                    
            self.optimizer.zero_grad()
        
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(stats_string.format(epoch+1, i, total_loss/(i+1)))
                total_loss = 0
    
    def train_model(self, train_iterator, dev_iterator, num_epocs = 30):
        self.model.train()
        min_loss = 1000000
        stats_string = "Epoch : {:2d} | Train Loss : {:4.4f} | Dev Loss : {:4.4f}"
        for i in range(num_epocs):
            self.train_epoch(i, train_iterator)
            dev_loss = self.evaluate(dev_iterator)
            train_loss = self.evaluate(train_iterator)
            print(stats_string.format(i+1, train_loss, dev_loss))
            if dev_loss <= min_loss:
                torch.save(self.model.state_dict(), 'best_model')
                min_loss = dev_loss
    
        print("Finished Training")
        
    def evaluate(self, iterator):
        self.model.eval()
        loss_total = 0
        
        for i, data in enumerate(iterator):
            inputs = data['image']
            labels = data['label'].long()
            
            if self.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            preds = self.model(inputs)
            loss = self.loss_fn(preds, labels)
            loss_total += loss.item()
        
        return loss_total
            


## === cell 9
model = CactusIdentifier()
print(model)


## === cell 10
trainer = TrainingModule(model)
trainer.train_model(train_loader, dev_loader, num_epocs = 1)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1097094845.py in <cell line: 0>()
      1 trainer = TrainingModule(model)
----> 2 trainer.train_model(train_loader, dev_loader, num_epocs = 1)

/tmp/ipykernel_11/3229608519.py in train_model(self, train_iterator, dev_iterator, num_epocs)
     42         stats_string = "Epoch : {:2d} | Train Loss : {:4.4f} | Dev Loss : {:4.4f}"
     43         for i in range(num_epocs):
---> 44             self.train_epoch(i, train_iterator)
     45             dev_loss = self.evaluate(dev_iterator)
     46             train_loss = self.evaluate(train_iterator)

/tmp/ipykernel_11/3229608519.py in train_epoch(self, epoch, train_iterator)
     16         total_loss = 0
     17         stats_string = "Epoch : {:3d} | Iteration : {:4d} | Loss : {:4.4f}"
---> 18         for i, data in enumerate(train_iterator):
     19             inputs = data['image']
     20             labels = data['label'].long()

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
     48         if self.auto_collation:
     49             if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
---> 50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
     52                 data = [self.dataset[idx] for idx in possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py in __getitems__(self, indices)
    418             return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
    419         else:
--> 420             return [self.dataset[self.indices[idx]] for idx in indices]
    421 
    422     def __len__(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py in <listcomp>(.0)
    418             return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
    419         else:
--> 420             return [self.dataset[self.indices[idx]] for idx in indices]
    421 
    422     def __len__(self):

/tmp/ipykernel_11/2581443546.py in __getitem__(self, idx)
     22         img_name = os.path.join(self.root_dir,
     23                                 self.cactus_annotations.iloc[idx, 0])
---> 24         image = io.imread(img_name)
     25         has_cactus = self.cactus_annotations.iloc[idx, 1]
     26 

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in fixed_func(*args, **kwargs)
    326                     kwargs[self.new_name] = deprecated_value
    327 
--> 328             return func(*args, **kwargs)
    329 
    330         if self.modify_docstring and func.__doc__ is not None:

/usr/local/lib/python3.11/dist-packages/skimage/io/_io.py in imread(fname, as_gray, plugin, **plugin_args)
     80 
     81     with file_or_url_context(fname) as fname, _hide_plugin_deprecation_warnings():
---> 82         img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
     83 
     84     if not hasattr(img, 'ndim'):

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in wrapped(*args, **kwargs)
    536             stacklevel = 1 + self.get_stack_length(func) - stack_rank
    537             warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
--> 538             return func(*args, **kwargs)
    539 
    540         # modify docstring to display deprecation warning

/usr/local/lib/python3.11/dist-packages/skimage/io/manage_plugins.py in call_plugin(kind, *args, **kwargs)
    252             raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')
    253 
--> 254     return func(*args, **kwargs)
    255 
    256 

/usr/local/lib/python3.11/dist-packages/skimage/io/_plugins/imageio_plugin.py in imread(*args, **kwargs)
      9 @wraps(imageio_imread)
     10 def imread(*args, **kwargs):
---> 11     out = np.asarray(imageio_imread(*args, **kwargs))
     12     if not out.flags['WRITEABLE']:
     13         out = out.copy()

/usr/local/lib/python3.11/dist-packages/imageio/v3.py in imread(uri, index, plugin, extension, format_hint, **kwargs)
     51         call_kwargs["index"] = index
     52 
---> 53     with imopen(uri, "r", **plugin_kwargs) as img_file:
     54         return np.asarray(img_file.read(**call_kwargs))
     55 

/usr/local/lib/python3.11/dist-packages/imageio/core/imopen.py in imopen(uri, io_mode, plugin, extension, format_hint, legacy_mode, **kwargs)
    111         request.format_hint = format_hint
    112     else:
--> 113         request = Request(uri, io_mode, format_hint=format_hint, extension=extension)
    114 
    115     source = "<bytes>" if isinstance(uri, bytes) else uri

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in __init__(self, uri, mode, extension, format_hint, **kwargs)
    247 
    248         # Parse what was given
--> 249         self._parse_uri(uri)
    250 
    251         # Set extension

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in _parse_uri(self, uri)
    407                 # Reading: check that the file exists (but is allowed a dir)
    408                 if not os.path.exists(fn):
--> 409                     raise FileNotFoundError("No such file: '%s'" % fn)
    410             else:
    411                 # Writing: check that the directory to write to does exist

FileNotFoundError: No such file: '/kaggle/input/aerial-cactus-identification/train/train/00b5670821357d6de4c9eec458b9da86.jpg'

## === cell 11
test_set = CactusImageDataset("../input/aerial-cactus-identification/sample_submission.csv", "../input/aerial-cactus-identification/test/test", image_transforms['test'])
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


## === cell 12
model.load_state_dict(torch.load('best_model'))
final_preds = []

for i, data in enumerate(test_loader):
    inputs = data['image']
    labels = data['label'].long()

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    preds = model(inputs)
    final_preds += preds[:,1].tolist()

test_set.cactus_annotations['has_cactus'] = final_preds
test_set.cactus_annotations.to_csv("samplesubmission.csv", index=False)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2065587013.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load('best_model'))
      2 final_preds = []
      3 
      4 for i, data in enumerate(test_loader):
      5     inputs = data['image']

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

FileNotFoundError: [Errno 2] No such file or directory: 'best_model'

## === cell 13
test_set.cactus_annotations.head()
