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

0.614293628808864

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import glob
import os.path as osp

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm.notebook import tqdm
from PIL import Image

import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

## === cell 3
df_train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
df_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")

## === cell 5
display(df_train)
display(df_train.labels.value_counts())

## === cell 6
d_set = set()
for k in df_train.labels.unique():
    d_set = d_set | set(k.split(" "))
print(f"num of labels: {len(d_set)}  {d_set}")

## === cell 8
display(df_sub)

## === cell 10
def to_label(df):
    """
    Function for Label encoding.
    """
    le = LabelEncoder()
    df["labels_n"] = le.fit_transform(df.labels.values)
    return df

df_train = to_label(df_train)
df_labels_idx = df_train.loc[df_train.duplicated(["labels", "labels_n"])==False]\
                [["labels_n", "labels"]].set_index("labels_n").sort_index()
display(df_labels_idx)

## === cell 12
def make_datapath_list(phase="train", val_size=0.25):
    """
    Function to create a PATH to the data.
    
    Parameters
    ----------
    phase : 'train' or 'val' or 'test'
        Specify whether to use Train data or test data.
    val_size : float
        Ratio of validation data to train data
        
    Returns
    -------
    path_lsit : list
        A list containing the PATH to the data.
    """
    
    if phase in ["train", "val"]:
        phase_path = "train_images"
    elif phase in ["test"]:
        phase_path = "test_images"
    else:
        print(f"{phase} not in path")
    rootpath = "/kaggle/input/plant-pathology-2021-fgvc8/"
    target_path = osp.join(rootpath+phase_path+"/*.jpg")
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
        
    if phase in ["train", "val"]:
        train, val = train_test_split(path_list, test_size=val_size, random_state=0, shuffle=True)
        if phase == "train":
            path_list = train
        else:
            path_list = val
    
    return path_list

## === cell 13
train_list = make_datapath_list(phase="train")
print(f"train data length : {len(train_list)}")
val_list = make_datapath_list(phase="val")
print(f"validation data length : {len(val_list)}")
test_list = make_datapath_list(phase="test")
print(f"test data length : {len(test_list)}")

## === cell 15
class ImageTransform():
    """
    Class for image preprocessing.
    
    Attributes
    ----------
    resize : int
        224
    mean : (R, G, B)
        Average value for each color channel
    std : (R, G, B)
        Standard deviation for each color channel
    """
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase="train"):
        """
        Parameters
        ----------
        phase: 'train' or 'val' or 'test'
            Specify the mode of preprocessing
        """
        return self.data_transform[phase](img)

## === cell 16
image_file_path = '/kaggle/input/plant-pathology-2021-fgvc8/train_images/800113bb65efe69e.jpg'
img = Image.open(image_file_path)

plt.imshow(img)
plt.show()

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = ImageTransform(size, mean, std)
img_transformed = transform(img, phase='train')
print(img_transformed.shape)

img_transformed = img_transformed.numpy().transpose([1, 2, 0])
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()

## === cell 17
class PlantDataset(data.Dataset):
    """
    Class to create a Dataset
    
    Attributes
    ----------
    df_train : DataFrame
        DataFrame containing the image labels.
    file_list : list
        A list containing the paths to the images
    transform : object
        Instance of the preprocessing class (ImageTransform)
    phase : 'train' or 'val' or 'test'
        Specify whether to use train, validation, or test
    """
    def __init__(self, df_train, file_list, transform=None, phase='train'):
        self.df_train = df_train
        self.df_labels_idx = df_labels_idx
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        """
        Returns the number of images.
        """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        Get data in Tensor format and labels of preprocessed images.
        """
        
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        image_name = img_path[-20:]
        
        if self.phase in ["train", "val"]:
            label = df_train.loc[df_train["image"]==image_name]["labels_n"].values[0]
        elif self.phase in ["test"]:
            label = -1
        
        return img_transformed, label, image_name

## === cell 18
train_dataset = PlantDataset(df_train, train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = PlantDataset(df_train, val_list, transform=ImageTransform(size, mean, std), phase='val')
test_dataset = PlantDataset(df_train, test_list, transform=ImageTransform(size, mean, std), phase='test')

index = 0

print("【train dataset】")
print(f"img num : {train_dataset.__len__()}")
print(f"img : {train_dataset.__getitem__(index)[0].size()}")
print(f"label : {train_dataset.__getitem__(index)[1]}")
print(f"image name : {train_dataset.__getitem__(index)[2]}")

print("\n【validation dataset】")
print(f"img num : {val_dataset.__len__()}")
print(f"img : {val_dataset.__getitem__(index)[0].size()}")
print(f"label : {val_dataset.__getitem__(index)[1]}")
print(f"image name : {val_dataset.__getitem__(index)[2]}")

print("\n【test dataset】")
print(f"img num : {test_dataset.__len__()}")
print(f"img : {test_dataset.__getitem__(index)[0].size()}")
print(f"label : {test_dataset.__getitem__(index)[1]}")
print(f"image name : {test_dataset.__getitem__(index)[2]}")

## === cell 20
batch_size = 128

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}



## === cell 22

use_pretrained = False
net = models.vgg16(pretrained=use_pretrained)


load_path = "/kaggle/input/d/kuboko/plantpathology2021/vgg16_pretrained.h"
if torch.cuda.is_available():
    load_weights = torch.load(load_path)
    net.load_state_dict(load_weights)
else:
    load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)

net.classifier[6] = nn.Linear(in_features=4096, out_features=12)

net.train()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2640451377.py in <cell line: 0>()
     13     net.load_state_dict(load_weights)
     14 else:
---> 15     load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
     16     net.load_state_dict(load_weights)
     17 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/d/kuboko/plantpathology2021/vgg16_pretrained.h'

## === cell 24
criterion = nn.CrossEntropyLoss()

## === cell 26
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ["features.24.weight", "features.24.bias", "features.26.weight", "features.26.bias", "features.28.weight", "features.28.bias"]
update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_1.append(param)
        print(f"Store in params_to_update_1 : {name}")
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print(f"Store in params_to_update_2 : {name}")
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print(f"Store in params_to_update_3 : {name}")
    else:
        param.requires_grad = False
        print(f"Parameters not to be learned :  {name}")

## === cell 27
optimizer = optim.SGD([
    {"params": params_to_update_1, "lr": 1e-4},
    {"params": params_to_update_2, "lr": 5e-4},
    {"params": params_to_update_3, "lr": 1e-3}
], momentum=0.9)

## === cell 29
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    """
    Function for training the model.
    
    Parameters
    ----------
    net: object
    dataloaders_dict: dictionary
    criterion: object
    optimizer: object
    num_epochs: int
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devices to be used : {device}")
    net.to(device)
    torch.backends.cudnn.benchmark = True
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")
        print("-------------------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            for inputs, labels, _ in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

## === cell 31
num_epochs = 4


## === cell 35
load_path = "/kaggle/input/d/kuboko/plantpathology2021/vgg16_fine_tuning_v1.h"
if torch.cuda.is_available():
    load_weights = torch.load(load_path)
    net.load_state_dict(load_weights)
else:
    load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/446969779.py in <cell line: 0>()
      4     net.load_state_dict(load_weights)
      5 else:
----> 6     load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
      7     net.load_state_dict(load_weights)

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/d/kuboko/plantpathology2021/vgg16_fine_tuning_v1.h'

## === cell 36
net

## === cell 39
class PlantPredictor():
    """
    Class for predicting labels from output results
    
    Attributes
    ----------
    df_labels_idx: DataFrame
        DataFrame that associates INDEX with a label name
    """
    
    def __init__(self, net, df_labels_idx, dataloaders_dict):
        self.net = net
        self.df_labels_idx = df_labels_idx
        self.dataloaders_dict = dataloaders_dict
        self.df_submit = pd.DataFrame()
        
    
    def __predict_max(self, out):
        """
        Get the label name with the highest probability.
        
        Parameters
        ----------
        predicted_label_name: str
            Name of the label with the highest prediction probability
        """
        maxid = np.argmax(out.detach().numpy(), axis=1)
        df_predicted_label_name = self.df_labels_idx.iloc[maxid]
        
        return df_predicted_label_name
    
    def inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Devices to be used : {device}")
        df_pred_list = []
        for inputs, _, image_name in tqdm(self.dataloaders_dict['test']):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(device)
            inputs = inputs.to(device)
            out = self.net(inputs)
            device = torch.device("cpu")
            out = out.to(device)
            df_pred = self.__predict_max(out).reset_index(drop=True)
            df_pred["image"] = image_name
            df_pred_list.append(df_pred)
            
        self.df_submit = pd.concat(df_pred_list, axis=0)
        self.df_submit = self.df_submit[["image", "labels"]].reset_index(drop=True)
            
        

## === cell 40
predictor = PlantPredictor(net, df_labels_idx, dataloaders_dict)
predictor.inference()



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_list_axis(self, key, axis)
   1713         try:
-> 1714             return self.obj._take_with_is_copy(key, axis=axis)
   1715         except IndexError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _take_with_is_copy(self, indices, axis)
   4152         """
-> 4153         result = self.take(indices=indices, axis=axis)
   4154         # Maybe set copy if we didn't actually change the index.

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in take(self, indices, axis, **kwargs)
   4132 
-> 4133         new_data = self._mgr.take(
   4134             indices,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in take(self, indexer, axis, verify)
    890         n = self.shape[axis]
--> 891         indexer = maybe_convert_indices(indexer, n, verify=verify)
    892 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexers/utils.py in maybe_convert_indices(indices, n, verify)
    281         if mask.any():
--> 282             raise IndexError("indices are out-of-bounds")
    283     return indices

IndexError: indices are out-of-bounds

The above exception was the direct cause of the following exception:

IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3138743817.py in <cell line: 0>()
      1 predictor = PlantPredictor(net, df_labels_idx, dataloaders_dict)
----> 2 predictor.inference()
      3 #df_pred = predictor.predict_max(out)
      4 
      5 #df_sub.labels = df_pred.labels.reset_index(drop=True)

/tmp/ipykernel_11/2423684457.py in inference(self)
     41             device = torch.device("cpu")
     42             out = out.to(device)
---> 43             df_pred = self.__predict_max(out).reset_index(drop=True)
     44             df_pred["image"] = image_name
     45             df_pred_list.append(df_pred)

/tmp/ipykernel_11/2423684457.py in __predict_max(self, out)
     26         """
     27         maxid = np.argmax(out.detach().numpy(), axis=1)
---> 28         df_predicted_label_name = self.df_labels_idx.iloc[maxid]
     29 
     30         return df_predicted_label_name

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1741         # a list of integers
   1742         elif is_list_like_indexer(key):
-> 1743             return self._get_list_axis(key, axis=axis)
   1744 
   1745         # a single integer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_list_axis(self, key, axis)
   1715         except IndexError as err:
   1716             # re-raise with different error message, e.g. test_getitem_ndarray_3d
-> 1717             raise IndexError("positional indexers are out-of-bounds") from err
   1718 
   1719     def _getitem_axis(self, key, axis: AxisInt):

IndexError: positional indexers are out-of-bounds

## === cell 41
df_submit = predictor.df_submit.copy()

## === cell 43
df_submit.to_csv("/kaggle/working/submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have an 'image' column.
