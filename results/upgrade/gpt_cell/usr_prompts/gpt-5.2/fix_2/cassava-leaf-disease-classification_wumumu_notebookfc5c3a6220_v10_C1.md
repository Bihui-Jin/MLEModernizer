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

3.9

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
def train(cur_epoch, dataloader, compute_grid=True):
    tq_description = "epoch %d" % cur_epoch
    tqbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    total_loss = 0.0
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

        total_loss += float(loss.detach().item())
        preds_list += preds.tolist()
        labels_list += labels.tolist()

    return preds_list, labels_list, total_loss


do_train(20)
generate_submission_csv()


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1730071010.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     32[0m [0;34m[0m[0m
[1;32m     33[0m [0mdo_train[0m[0;34m([0m[0;36m20[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m [0mgenerate_submission_csv[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/2184603836.py[0m in [0;36mgenerate_submission_csv[0;34m()[0m
[1;32m     36[0m     [0mnames_list[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m     [0mpreds_list[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 38[0;31m     [0;32mfor[0m [0mi[0m[0;34m,[0m [0mitem[0m [0;32min[0m [0mtqbar[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m         [0mtqbar[0m[0;34m.[0m[0mset_description[0m[0;34m([0m[0mtq_description[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m         [0mimages[0m[0;34m,[0m [0mnames[0m [0;34m=[0m [0mitem[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tqdm/std.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m   1179[0m [0;34m[0m[0m
[1;32m   1180[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1181[0;31m             [0;32mfor[0m [0mobj[0m [0;32min[0m [0miterable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1182[0m                 [0;32myield[0m [0mobj[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1183[0m                 [0;31m# Update and possibly print the progressbar.[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/ipykernel_11/2699512916.py[0m in [0;36m__getitem__[0;34m(self, index)[0m
[1;32m     32[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mindex[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m         [0mimage_name[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mimage_list[0m[0;34m[[0m[0mindex[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m         [0mimage[0m [0;34m=[0m [0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdata_path[0m[0;34m,[0m [0mimage_name[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     35[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mtransforms[0m[0;34m:[0m [0mimage[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtransforms[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m         [0;32mreturn[0m [0mimage[0m[0;34m,[0m [0mimage_name[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/PIL/Image.py[0m in [0;36mopen[0;34m(fp, mode, formats)[0m
[1;32m   3511[0m     [0;32mif[0m [0mis_path[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3512[0m         [0mfilename[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mfspath[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3513[0;31m         [0mfp[0m [0;34m=[0m [0mbuiltins[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3514[0m         [0mexclusive_fp[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3515[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIsADirectoryError[0m: [Errno 21] Is a directory: '/kaggle/input/cassava-leaf-disease-classification/test_images/test_images'
