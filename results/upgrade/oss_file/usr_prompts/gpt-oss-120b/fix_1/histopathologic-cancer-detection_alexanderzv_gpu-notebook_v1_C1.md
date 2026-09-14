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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.761

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from typing import List
import logging
from typing import Optional
from functools import partial
from typing import Tuple
from typing import Union


import torch.nn as nn
import numpy as np
import os
import pandas as pd
import torch
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchvision import transforms
from torch.autograd import Variable
import albumentations as A


## === cell 1
DATA_FOLDER = '../input/histopathologic-cancer-detection'
LABELS = f'{DATA_FOLDER}/train_labels.csv'
TRAIN_IMAGES_FOLDER = f'{DATA_FOLDER}/train'
SAMPLE_SUBMISSION = f'{DATA_FOLDER}/sample_submission.csv'
USE_GPU = torch.cuda.is_available()


## === cell 2
USE_GPU


## === cell 3
logging.basicConfig(level='INFO')
logger = logging.getLogger()


## === cell 4
labels = pd.read_csv(LABELS)


## === cell 5
labels


## === cell 6
def format_labels_for_data_set(labels):
    return (labels['label'].values.reshape(-1,1))

def train_valid_split(df, split_percent, limit_df= 10000 ):
    df = df.sample(n = df.shape[0])
    df = df.iloc[:limit_df]
    split = round(limit_df * split_percent / 100)
    train = df.iloc[:split]
    valid = df.iloc[split:]
    return (train, valid)

def format_path_to_images_for_dataset(labels, path):
    return [os.path.join(path, f'{f}.tif') for f in labels['id'].values]


## === cell 7
class MainDataset(Dataset):
    def __init__(self, x_dataset, y_dataset, x_tfms):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_tfms = x_tfms
        
    def __len__(self):
        return self.x_dataset.__len__() 
        
    def __getitem__(self, index):
        x = self.x_dataset[index]
        y = self.y_dataset[index]
        if x_tfms is not None:
            x = self.x_tfms(x)
        return x, y

class ImageDataset(Dataset):
    def __init__(self, path_to_image):
        self.path_to_image = path_to_image
    
    def __len__(self):
        return len(self.path_to_image)
    
    def __getitem__(self, index):
        img = Image.open(self.path_to_image[index])
        
        augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images
            A.OneOf(
                [
                    A.RandomContrast(), # apply random contrast
                    A.RandomGamma(), # apply random gamma
                    A.RandomBrightness(limit = -0.1), # apply random brightness
                ],
                p = 1
            ),
            A.ShiftScaleRotate(p = 0.5)
        ],
        p = 1)
        
        image_aug = augmentation_pipeline(image = np.array(img))['image']
        image = Image.fromarray(image_aug, 'RGB')
        return image


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
    
    def __len__(self):
        return len(labels)
    
    def __getitem__(self, index):
        return self.labels[index]


## === cell 8
labels = pd.read_csv(LABELS)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION)

train, valid = train_valid_split(labels, 70)

train_labels = format_labels_for_data_set(train)
valid_labels = format_labels_for_data_set(valid)

train_images = format_path_to_images_for_dataset(train, TRAIN_IMAGES_FOLDER)
valid_images = format_path_to_images_for_dataset(valid, TRAIN_IMAGES_FOLDER)

train_images_dataset = ImageDataset(train_images)
valid_images_dataset = ImageDataset(valid_images)
train_labels_dataset = LabelDataset(train_labels)
valid_labels_dataset = LabelDataset(valid_labels)


## === cell 9
def implot(dataset, w=2, h=2, cols=12, max_charts = 24 ):
    rows = (max_charts) / cols + 1
    images = [dataset[3] for i in range(max_charts)]
    plt.figure(figsize = (cols * w, rows * h))
    plt.tight_layout()
    for chart, img in enumerate(images, 1):
        ax = plt.subplot(rows, cols, chart)
        ax.imshow(np.array(img))
        ax.axis('off')


## === cell 10
implot(train_images_dataset)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2569946566.py in <cell line: 0>()
----> 1 implot(train_images_dataset)

/tmp/ipykernel_11/3682135001.py in implot(dataset, w, h, cols, max_charts)
      2 def implot(dataset, w=2, h=2, cols=12, max_charts = 24 ):
      3     rows = (max_charts) / cols + 1
----> 4     images = [dataset[3] for i in range(max_charts)]
      5     plt.figure(figsize = (cols * w, rows * h))
      6     plt.tight_layout()

/tmp/ipykernel_11/3682135001.py in <listcomp>(.0)
      2 def implot(dataset, w=2, h=2, cols=12, max_charts = 24 ):
      3     rows = (max_charts) / cols + 1
----> 4     images = [dataset[3] for i in range(max_charts)]
      5     plt.figure(figsize = (cols * w, rows * h))
      6     plt.tight_layout()

/tmp/ipykernel_11/567410299.py in __getitem__(self, index)
     33                 [
     34                     # apply one of transforms to 50% of images
---> 35                     A.RandomContrast(), # apply random contrast
     36                     A.RandomGamma(), # apply random gamma
     37                     A.RandomBrightness(limit = -0.1), # apply random brightness

AttributeError: module 'albumentations' has no attribute 'RandomContrast'

## === cell 11
x_tfms = transforms.Compose([transforms.ToTensor(), 
                             transforms.Normalize(
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]
                             )
                            ])


## === cell 12
train_dataset = MainDataset(train_images_dataset, train_labels_dataset, x_tfms)
valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset, x_tfms)


## === cell 14
shuffle = True
batch_size = 512
num_workers = 0

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)


## === cell 16
def to_gpu(tensor):
    return tensor.cuda() if USE_GPU else tensor

def create_resnet9_model(output_dim: int = 1) -> nn.Module:
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    in_features = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(in_features, output_dim)
    model = to_gpu(model)
    return model


## === cell 17
resnet9 = create_resnet9_model()
resnet9


## === cell 18
lr = 1e-3
optimizer = Adam(resnet9.parameters(), lr)


## === cell 19
loss = nn.BCEWithLogitsLoss()


## === cell 20
def auc_writer(y_true, y_predicted, iteration):
    try:
        score = roc_auc_score(np.vstack(y_true), np.vstack(y_predicted))
    except:
        score = -1
    print(f'iteration: {iteration}, roc_auc: {score}')
    logger.info(f'iteration: {iteration}, roc_auc: {score}')    
    
loss_writer_train = auc_writer
loss_writer_valid = auc_writer


## === cell 22
def predict(model, dataloader):
    model.eval()
    y_true, y_hat = [], []
    
    for x, y in dataloader:
        x = Variable(T(x))
        y = Variable(T(y))
        output = model(x)
        
        y_true.append(to_numpy(y))
        y_hat.append(to_numpy(output))
    
    return y_true, y_hat


## === cell 23
def T(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.FloatTensor(tensor)
    else:
        tensor = tensor.type(torch.FloatTensor)
    if USE_GPU:
        tensor = to_gpu(tensor)
    return tensor


def to_numpy(tensor):
    if type(tensor) == np.array or type(tensor) == np.ndarray:
        return np.array(tensor)
    elif type(tensor) == Image.Image:
        return np.array(tensor)
    elif type(tensor) == Tensor:
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError(msg)


## === cell 24
def iteration_trigger(iteration, every_x_iteration):
    if every_x_iteration == 1:
        return True
    elif iteration > 0 and iteration % every_x_iteration == 0:
        return True
    else:
        return False
    


def init_triggers(step = 1, train = 10, valid = 10):
    do_step_trigger = partial(iteration_trigger, every_x_iteration = step)
    train_loss_trigger = partial(iteration_trigger, every_x_iteration = train)
    valid_loss_trigger = partial(iteration_trigger, every_x_iteration = valid)
    
    return do_step_trigger, train_loss_trigger, valid_loss_trigger


do_step_trigger, train_loss_trigger, valid_loss_trigger = init_triggers(1, 10, 20)


## === cell 25
def train_one_epoch(model, 
                    train_data_loader, 
                    valid_data_loader, 
                    loss, 
                    optimizer, 
                    loss_writer_train, 
                    loss_writer_valid,
                    do_step_trigger,
                    train_loss_trigger,
                    valid_loss_trigger):
    
    y_true_train, y_hat_train = [], []
    for iteration, (x, y) in enumerate(train_data_loader):
        x_train = Variable(T(x), requires_grad = True)
        y_train = Variable(T(y), requires_grad = True)
        
        output = model(x_train)
        y_true_train.append(to_numpy(y_train))
        y_hat_train.append(to_numpy(output))
        loss_values = loss(output, y_train)
        loss_values.backward()
        
        if do_step_trigger(iteration):
            optimizer.step()
            optimizer.zero_grad()
        
        if train_loss_trigger(iteration):
            print('train_loss_trigger: ')
            loss_writer_train(y_true_train, y_hat_train, iteration)
            y_true_train, y_hat_train = [], []
        
        if valid_loss_trigger(iteration):
            print('valid_loss_trigger:')
            y_true_valid, y_hat_valid = predict(model, valid_data_loader)
            loss_writer_valid(y_true_valid, y_hat_valid, iteration)
        
    return model


## === cell 26
resnet9 = train_one_epoch(resnet9, 
                    train_dataloader, 
                    valid_dataloader, 
                    loss, 
                    optimizer, 
                    loss_writer_train, 
                    loss_writer_valid,
                    do_step_trigger,
                    train_loss_trigger,
                    valid_loss_trigger)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2904197306.py in <cell line: 0>()
----> 1 resnet9 = train_one_epoch(resnet9, 
      2                     train_dataloader,
      3                     valid_dataloader,
      4                     loss,
      5                     optimizer,

/tmp/ipykernel_11/4106902128.py in train_one_epoch(model, train_data_loader, valid_data_loader, loss, optimizer, loss_writer_train, loss_writer_valid, do_step_trigger, train_loss_trigger, valid_loss_trigger)
     11 
     12     y_true_train, y_hat_train = [], []
---> 13     for iteration, (x, y) in enumerate(train_data_loader):
     14         x_train = Variable(T(x), requires_grad = True)
     15         y_train = Variable(T(y), requires_grad = True)

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
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/567410299.py in __getitem__(self, index)
     10 
     11     def __getitem__(self, index):
---> 12         x = self.x_dataset[index]
     13         y = self.y_dataset[index]
     14         if x_tfms is not None:

/tmp/ipykernel_11/567410299.py in __getitem__(self, index)
     33                 [
     34                     # apply one of transforms to 50% of images
---> 35                     A.RandomContrast(), # apply random contrast
     36                     A.RandomGamma(), # apply random gamma
     37                     A.RandomBrightness(limit = -0.1), # apply random brightness

AttributeError: module 'albumentations' has no attribute 'RandomContrast'

## === cell 27
TEST_IMAGES_FOLDER = f'{DATA_FOLDER}/test/'


def test_image_collection(directory: str) -> List:
    images_name = []
    for filename in os.listdir(directory):
        images_name.append(TEST_IMAGES_FOLDER + filename)
    return(images_name)

test_image = test_image_collection(TEST_IMAGES_FOLDER)
test_images_dataset = ImageDataset(test_image)    

class TestDataset(Dataset):
    def __init__(self, x_dataset: Dataset, x_tfms: Optional = None):
        self.x_dataset = x_dataset
        self.x_tfms = x_tfms
        
    def __len__(self) -> int:
        return self.x_dataset.__len__() 
        
    def __getitem__(self, index: int) -> Tuple:
        x = self.x_dataset[index]
        if x_tfms is not None:
            x = self.x_tfms(x)
        return x
    
test_dataset = TestDataset(test_images_dataset, x_tfms)    

batch_size = 512
num_workers = 0
shuffle = False

test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)


## === cell 28
def predict_test(model, dataloader):
    model.eval()
    y_hat = []
    
    for x in dataloader:
        x = Variable(T(x))
        output = model(x)
        
        y_hat.append(to_numpy(output))
    return y_hat


## === cell 29
y_hat_test = predict_test(resnet9, test_dataloader)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2044234507.py in <cell line: 0>()
      1 # сделаем предсказания для тестовой выборки
----> 2 y_hat_test = predict_test(resnet9, test_dataloader)

/tmp/ipykernel_11/2225971200.py in predict_test(model, dataloader)
      4     y_hat = []
      5 
----> 6     for x in dataloader:
      7         x = Variable(T(x))
      8         output = model(x)

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
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/3812584363.py in __getitem__(self, index)
     23 
     24     def __getitem__(self, index: int) -> Tuple:
---> 25         x = self.x_dataset[index]
     26         if x_tfms is not None:
     27             x = self.x_tfms(x)

/tmp/ipykernel_11/567410299.py in __getitem__(self, index)
     33                 [
     34                     # apply one of transforms to 50% of images
---> 35                     A.RandomContrast(), # apply random contrast
     36                     A.RandomGamma(), # apply random gamma
     37                     A.RandomBrightness(limit = -0.1), # apply random brightness

AttributeError: module 'albumentations' has no attribute 'RandomContrast'

## === cell 30
predictions = pd.DataFrame(
    list(
        zip(
            test_image,
            np.vstack(y_hat_test).reshape(-1)
        )
    ), 
     columns=['id', 'label'])
predictions['id'] = predictions['id'].apply(lambda x: x.split('/')[-1].split('.')[0]) 


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1960689287.py in <cell line: 0>()
      4         zip(
      5             test_image,
----> 6             np.vstack(y_hat_test).reshape(-1)
      7         )
      8     ), 

NameError: name 'y_hat_test' is not defined

## === cell 31
predictions.to_csv('submission.csv', index=False)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1817883175.py in <cell line: 0>()
----> 1 predictions.to_csv('submission.csv', index=False)

NameError: name 'predictions' is not defined

## === cell 32
predictions


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4129765282.py in <cell line: 0>()
----> 1 predictions

NameError: name 'predictions' is not defined

## === cell 33
max_charts = 60
def implot_errors(files, w=2, h=2, cols=12):
    rows = len(files) / cols + 1
    images = [Image.open(f) for f in files]
    plt.figure(figsize = (cols * w, rows * h))
    plt.tight_layout()
    for chart, img in enumerate(images, 1):
        ax = plt.subplot(rows, cols, chart)
        ax.imshow(np.array(img))
        ax.axis('off')


## === cell 34
y_true, y_hat = predict(resnet9, valid_dataloader)

predictions_comparison = pd.DataFrame(
    list(
        zip(
            valid_labels.reshape(-1), 
            np.vstack(y_hat).reshape(-1),
            valid_images
        )
    ), 
     columns=['true', 'pred', 'files'])

predictions_comparison.head(3)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1468237039.py in <cell line: 0>()
      1 # сделаем таблицу с предсказаниями и реальными значениями на валидационной выборке
----> 2 y_true, y_hat = predict(resnet9, valid_dataloader)
      3 
      4 predictions_comparison = pd.DataFrame(
      5     list(

/tmp/ipykernel_11/3164989031.py in predict(model, dataloader)
      3     y_true, y_hat = [], []
      4 
----> 5     for x, y in dataloader:
      6         x = Variable(T(x))
      7         y = Variable(T(y))

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
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/567410299.py in __getitem__(self, index)
     10 
     11     def __getitem__(self, index):
---> 12         x = self.x_dataset[index]
     13         y = self.y_dataset[index]
     14         if x_tfms is not None:

/tmp/ipykernel_11/567410299.py in __getitem__(self, index)
     33                 [
     34                     # apply one of transforms to 50% of images
---> 35                     A.RandomContrast(), # apply random contrast
     36                     A.RandomGamma(), # apply random gamma
     37                     A.RandomBrightness(limit = -0.1), # apply random brightness

AttributeError: module 'albumentations' has no attribute 'RandomContrast'

## === cell 35
files = predictions_comparison[predictions_comparison['true']==1].sort_values('pred')['files'].values[:max_charts]
implot_errors(files)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2268862651.py in <cell line: 0>()
----> 1 files = predictions_comparison[predictions_comparison['true']==1].sort_values('pred')['files'].values[:max_charts]
      2 implot_errors(files)

NameError: name 'predictions_comparison' is not defined

## === cell 36
files = predictions_comparison[predictions_comparison['true']==0].sort_values('pred', ascending=False)['files'].values[:max_charts]
implot_errors(files)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1642080911.py in <cell line: 0>()
----> 1 files = predictions_comparison[predictions_comparison['true']==0].sort_values('pred', ascending=False)['files'].values[:max_charts]
      2 implot_errors(files)

NameError: name 'predictions_comparison' is not defined

## === cell 37
files = predictions_comparison[predictions_comparison['true']==1].sort_values('pred', ascending=False)['files'].values[:max_charts]
implot_errors(files)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2127841041.py in <cell line: 0>()
----> 1 files = predictions_comparison[predictions_comparison['true']==1].sort_values('pred', ascending=False)['files'].values[:max_charts]
      2 implot_errors(files)

NameError: name 'predictions_comparison' is not defined

## === cell 38
files = predictions_comparison[predictions_comparison['true']==0].sort_values('pred', ascending=True)['files'].values[:max_charts]
implot_errors(files)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/449796607.py in <cell line: 0>()
----> 1 files = predictions_comparison[predictions_comparison['true']==0].sort_values('pred', ascending=True)['files'].values[:max_charts]
      2 implot_errors(files)

NameError: name 'predictions_comparison' is not defined
