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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.5945

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import json
import os
import tqdm
import time
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models


map_json_path = '/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json'
train_data_path = '/kaggle/input/cassava-leaf-disease-classification/train_images/'
test_data_path = '/kaggle/input/cassava-leaf-disease-classification/test_images/'
train_csv_path = '/kaggle/input/cassava-leaf-disease-classification/train.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_pretrained = False
batch_size = 8
img_resize = (100, 100)


## === cell 1
class get_dataset(Dataset):
    def __init__(self, data_path, csv_label_path, train, train_size, transforms=None):
        self.train = train
        self.data_path = data_path
        self.transforms = transforms
        
        self.label_csv = pd.read_csv(csv_label_path)
        self.label_dict = self.label_csv.set_index('image_id')['label'].to_dict()
       
        images_name_list = os.listdir(data_path)[:20]
        train_image, test_image = train_test_split(images_name_list, train_size=train_size, random_state=0)
        self.image_list = train_image if self.train else test_image
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.label_dict[image_name]
        image = Image.open(os.path.join(self.data_path, image_name))
        
        if self.transforms: image = self.transforms(image)
        return image, label
    
    def __len__(self):
        return len(self.image_list)
    
    
class get_test_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.image_list = os.listdir(data_path)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(os.path.join(self.data_path, image_name))
        if self.transforms: image = self.transforms(image)
        return image, image_name
    
    def __len__(self):
        return len(self.image_list)   
    

mytransforms = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(),
])


train_dataset = get_dataset(train_data_path, train_csv_path, True, 0.9, mytransforms)
validation_dataset = get_dataset(train_data_path, train_csv_path, False, 0.9, mytransforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

test_dataset = get_test_dataset(test_data_path, mytransforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


## === cell 2
with open(map_json_path, 'r') as f:
    map_json_data = json.load(f)

map_json_data


## === cell 3
plt.figure(figsize=(20, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    
    image = train_dataset[i][0]
    image = image*0.5 + 0.5
    image = transforms.ToPILImage()(image)
    
    plt.title(map_json_data[str(train_dataset[i][1])])
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([]) 


## === cell 4
model = models.vgg16_bn(pretrained=model_pretrained)
sequential = list(model.classifier[:3])
sequential.append(nn.Linear(4096, 5))
model.classifier = nn.Sequential(*sequential)
model.to(device)
    
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


## === cell 5
def train(cur_epoch, dataloader, compute_grid=True):
    tq_description = 'epoch %d'%cur_epoch
    tqbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    total_loss = 0
    preds_list = []
    labels_list = []

    for i, item in tqbar:
        tqbar.set_description(tq_description)
        images, labels = item
        images = images.to(device)
        labels = labels.to(device)
            
        model_out = model(images)
        loss = criterion(model_out, labels)
        _, preds = torch.max(model_out, 1)
        
        if compute_grid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss
        preds_list += preds.tolist()
        labels_list += labels.tolist()
        
    return preds_list, labels_list, total_loss


def generate_submission_csv():
    tq_description = 'generate csv'
    tqbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    model.load_state_dict(torch.load('model.pkl'))
    
    names_list = []
    preds_list = []
    for i, item in tqbar:
        tqbar.set_description(tq_description)
        images, names = item
        
        images = images.to(device)
        model_out = model(images)
        _, preds = torch.max(model_out, 1)
        
        names_list += list(names)
        preds_list += preds.tolist()
        
    submission = pd.DataFrame({ 'image_id': names_list, 'label': preds_list })
    submission.to_csv("submission.csv", index=False)


## === cell 6
def compute_recall(preds_list, labels_list, class_num):
    recall_arr = np.zeros(class_num)
    
    for i in range(class_num):
        i_labels_list = labels_list == i
        i_preds_list = preds_list == i 
        total_i_class_num = np.sum(i_labels_list)
        preds_i_class_num = np.sum(i_preds_list * i_labels_list)
        recall_arr[i] = preds_i_class_num / total_i_class_num if total_i_class_num != 0 else 0
    return recall_arr
    
    
def compute_accuracy(preds_list, labels_list):
    return np.sum(preds_list == labels_list) / len(labels_list)

    
def do_train(epoch):       
    train_loss_list = []
    train_accuracy_list = []
    train_recall_list = []
    
    val_loss_list = []
    val_accuracy_list = []
    val_recall_list = []
    
    best_accuracy = [-1 ,-1] #(epoch, value)
    train_image_num = len(train_dataset)
    val_image_num = len(test_dataset)
    
    print('info:')
    print('train image number: ', train_image_num)
    print('validation image number:', val_image_num)
    print('train on: %s'%device)
    print('train epoch: %d'%epoch)
    
    for i in range(epoch):
        preds_list, labels_list, total_loss = train(i, train_dataloader, True)
        accuracy = compute_accuracy(preds_list, labels_list)
        recall = compute_recall(preds_list, labels_list, 5)
        train_loss_list.append(total_loss)
        train_accuracy_list.append(accuracy)
        train_recall_list.append(recall)
        print('train loss: %f'%total_loss)
        print('train accuracy: %f'%accuracy)
        print('train recall:', recall)
        
        preds_list, labels_list, total_loss = train(i, validation_dataloader, False)
        accuracy = compute_accuracy(preds_list, labels_list)
        recall = compute_recall(preds_list, labels_list, 5)
        val_loss_list.append(total_loss)
        val_accuracy_list.append(accuracy)
        val_recall_list.append(recall)
        print('test loss: %f'%total_loss)
        print('test accuracy: %f'%accuracy)
        print('test recall:', recall)
        
        if best_accuracy[1] < accuracy:
            best_accuracy[0] = epoch
            best_accuracy[1] = accuracy
            torch.save(model.state_dict(), 'model.pkl')
            
    plt.figure()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.title('loss')
    plt.legend(labels=['train','validation'])
    
    plt.figure()
    plt.plot(train_accuracy_list)
    plt.plot(val_accuracy_list)
    plt.title('accuracy')
    plt.legend(labels=['train','validation'])
    
    plt.figure()
    plt.plot(train_recall_list)
    plt.plot(val_recall_list)
    plt.title('recall')
    plt.legend(labels=['train','validation'])


## === cell 7
do_train(20)
generate_submission_csv()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2587243340.py in <cell line: 0>()
----> 1 do_train(20)
      2 generate_submission_csv()

/tmp/ipykernel_11/2227812874.py in do_train(epoch)
     62     # plot data
     63     plt.figure()
---> 64     plt.plot(train_loss_list)
     65     plt.plot(val_loss_list)
     66     plt.title('loss')

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
   2810 @_copy_docstring_and_deprecators(Axes.plot)
   2811 def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
-> 2812     return gca().plot(
   2813         *args, scalex=scalex, scaley=scaley,
   2814         **({"data": data} if data is not None else {}), **kwargs)

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py in plot(self, scalex, scaley, data, *args, **kwargs)
   1686         """
   1687         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
-> 1688         lines = [*self._get_lines(*args, data=data, **kwargs)]
   1689         for line in lines:
   1690             self.add_line(line)

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_base.py in __call__(self, data, *args, **kwargs)
    309                 this += args[0],
    310                 args = args[1:]
--> 311             yield from self._plot_args(
    312                 this, kwargs, ambiguous_fmt_datakey=ambiguous_fmt_datakey)
    313 

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_base.py in _plot_args(self, tup, kwargs, return_kwargs, ambiguous_fmt_datakey)
    494             y = _check_1d(xy[1])
    495         else:
--> 496             x, y = index_of(xy[-1])
    497 
    498         if self.axes.xaxis is not None:

/usr/local/lib/python3.11/dist-packages/matplotlib/cbook/__init__.py in index_of(y)
   1659         pass
   1660     try:
-> 1661         y = _check_1d(y)
   1662     except (np.VisibleDeprecationWarning, ValueError):
   1663         # NumPy 1.19 will warn on ragged input, and we can't actually use it.

/usr/local/lib/python3.11/dist-packages/matplotlib/cbook/__init__.py in _check_1d(x)
   1351             not hasattr(x, 'ndim') or
   1352             len(x.shape) < 1):
-> 1353         return np.atleast_1d(x)
   1354     else:
   1355         return x

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in atleast_1d(*arys)
     63     res = []
     64     for ary in arys:
---> 65         ary = asanyarray(ary)
     66         if ary.ndim == 0:
     67             result = ary.reshape(1)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in __array__(self, dtype)
   1192             return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
   1193         if dtype is None:
-> 1194             return self.numpy()
   1195         else:
   1196             return self.numpy().astype(dtype, copy=False)

TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
