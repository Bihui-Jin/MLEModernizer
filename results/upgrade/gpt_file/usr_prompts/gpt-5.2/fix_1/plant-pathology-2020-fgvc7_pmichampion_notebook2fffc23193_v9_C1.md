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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.5106922822571176

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


import torch
import torch.utils.data as Data
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import softmax
import cv2
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from tqdm.notebook import tqdm
from albumentations import *
from albumentations.pytorch import ToTensorV2
import gc
import os

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2571220858.py in <cell line: 0>()
     12 import cv2
     13 from transformers import get_cosine_schedule_with_warmup
---> 14 from transformers import AdamW
     15 from tqdm.notebook import tqdm
     16 from albumentations import *

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 2
fig, axs = plt.subplots(2,2, figsize=(14,10))
im_healthy = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_2.jpg')
im_multi = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_1.jpg')
im_rust = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_3.jpg')
im_scab = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')

plt.subplot(2,2,1)
plt.imshow(im_healthy)

plt.subplot(2,2,2)
plt.imshow(im_multi)

plt.subplot(2,2,3)
plt.imshow(im_rust)

plt.subplot(2,2,4)
plt.imshow(im_scab)

## === cell 4
Img_folder = '/kaggle/input/plant-pathology-2020-fgvc7/images/'

def get_path_of_img(filename):
    return Img_folder + filename + '.jpg'

train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')

train.head()

## === cell 5
train['image_path'] = train['image_id'].apply(get_path_of_img)
test['image_path'] = test['image_id'].apply(get_path_of_img)

## === cell 6
from sklearn.model_selection import train_test_split

train_targets = train.loc[:, 'healthy':'scab']
train_paths = train.image_path
test_paths = test.image_path

## === cell 7
train_paths, valid_paths, train_targets, valid_targets = train_test_split(train_paths, train_targets,
                                                                         test_size=0.2, random_state=27,
                                                                         shuffle=True, stratify=train_targets)


## === cell 8
train_paths.head()

## === cell 9
train_targets.head()

## === cell 10
train_targets.head()

## === cell 11
train_paths.head()

## === cell 13
img_scab = plt.imread(train_paths[3])
plt.subplot(1,1, 1)
plt.imshow(img_scab)

## === cell 15
class Leaf_Dataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, test=False, train=True):
        self.paths = image_paths
        self.test = test
        if self.test == False:
            self.labels = labels
        self.train = train
        self.train_transform = Compose([ 
            HorizontalFlip(p = 0.5), 
            VerticalFlip(p = 0.5),
            ShiftScaleRotate(rotate_limit=25.0, p=0.7),
            RandomBrightnessContrast(p = 0.7, brightness_limit = 0.2, contrast_limit = 0.2),
            OneOf([
                Sharpen(p = 1), #либо повысим резкость
                Blur(p = 1) # либо превратим картинку в мыло
            ], p = 0.5),
            Resize(height=224, width=224)
        ])
        self.test_transform = Compose([
            HorizontalFlip(p = 0.5),
            VerticalFlip(p = 0.5),
            ShiftScaleRotate(rotate_limit=25.0, p=0.7),
            Resize(height=224, width=224)
        ])
        self.default_transform = Compose([
            Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                always_apply=True),
                ToTensorV2()
        ])
    
    def __len__(self):
        return self.paths.shape[0]
    
    
    def __getitem__(self, item):
        img = cv2.imread(self.paths[item])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.test is False:
            label = torch.tensor(np.argmax(self.labels.loc[item].values))
        if self.train is True:
            img = self.train_transform(image=img)['image']
            img = self.default_transform(image=img)['image']
        
        elif self.test is True:
            img= self.test_transform(image=img)['image']
            img= self.default_transform(image=img)['image']
        
        else:
            img = self.default_transform(image=img)['image']
        
        if self.test == False:
            return img, label
        return img
        

## === cell 17
def train_fn(net, loader):
    running_loss = 0
    model_predictions = []
    accuracy_labels = []
    pbar = tqdm(total = len(loader), desc='Training') # прогресс бар для обучения
    net.train()
    for _, (images, labels) in enumerate(loader):
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad() # обнуляем градиент перед обратным проходом
        predictions = net(images) # предсказываем
        loss = loss_fn(predictions, labels) # вычисляем лосс
        loss.backward() # считаем градиенты
        optimizer.step() # переопределяем их по градиентам
        scheduler.step() # изменяем шаг для градиентного спуска
        
        running_loss += loss.item() * labels.shape[0] # умножаем потерю на количество в батче
        accuracy_labels = np.concatenate((accuracy_labels, labels.cpu().numpy()), 0)
        model_predictions = np.concatenate((model_predictions, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
        pbar.update()
    accuracy = accuracy_score(accuracy_labels, model_predictions)
    pbar.close()
    return running_loss / CFG.train_size, accuracy


def valid_fn(net, loader):
    
    running_loss = 0
    model_predictions = []
    accuracy_labels = []
    pbar = tqdm(total=len(loader), desc='Validation')
    net.eval()
    with torch.no_grad(): # для валидации нет необходимости в подсчете градиентов
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            predictions = net(images) # предсказываем
            loss = loss_fn(predictions, labels) # вычисляем лосс

            running_loss += loss.item() * labels.shape[0] # умножаем потерю на количество в батче
            accuracy_labels = np.concatenate((accuracy_labels, labels.cpu().numpy()), 0)
            model_predictions = np.concatenate((model_predictions, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
            pbar.update()
            
        accuracy = accuracy_score(accuracy_labels, model_predictions)
        conf_matrix = confusion_matrix(accuracy_labels, model_predictions)
        
    pbar.close()
    return running_loss / CFG.valid_size, accuracy, conf_matrix


def test_fn(net, loader):
    
    preds_for_output = np.zeros((1,4)) # у нас четыре возможных класса
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total = len(loader), desc='Train')
        for _, images in enumerate(loader):
            images = images.to(CFG.device)
            predictions = net(images)
            preds_for_output = np.concatenate((preds_for_output, predictions.cpu().detach().numpy()), 0)
            pbar.update()
    
    pbar.close()
    return preds_for_output

## === cell 18
class CFG:
    batch_size = 8
    num_epochs = 30
    train_size = train_targets.shape[0]
    valid_size = valid_targets.shape[0]
    model_name = 'ResNet18'
    device = 'cuda'
    lr = 8e-4

## === cell 19
train_targets.reset_index(drop=True, inplace=True)
train_paths.reset_index(drop=True, inplace=True)
valid_targets.reset_index(drop=True, inplace=True)
valid_paths.reset_index(drop=True, inplace=True)

## === cell 20
len(train_paths)

## === cell 22

train_dataset = Leaf_Dataset(train_paths, labels = train_targets, test = False, train = True)
train_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers = 2)

valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, train = False, test=False)
valid_loader = Data.DataLoader(valid_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers = 2)

test_dataset = Leaf_Dataset(test_paths, labels = train_targets, test = True, train = False)
test_loader = Data.DataLoader(test_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers = 2)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/428768208.py in <cell line: 0>()
----> 1 train_dataset = Leaf_Dataset(train_paths, labels = train_targets, test = False, train = True)
      2 train_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers = 2)
      3 
      4 valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, train = False, test=False)
      5 valid_loader = Data.DataLoader(valid_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers = 2)

/tmp/ipykernel_11/2167977243.py in __init__(self, image_paths, labels, test, train)
      7         self.train = train
      8         # устанавливаем pipeline трансформаций для изображений train
----> 9         self.train_transform = Compose([ 
     10             HorizontalFlip(p = 0.5),
     11             VerticalFlip(p = 0.5),

NameError: name 'Compose' is not defined

## === cell 23
len(train_dataset)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3778854998.py in <cell line: 0>()
----> 1 len(train_dataset)

NameError: name 'train_dataset' is not defined

## === cell 24
train_paths

## === cell 25
from torchvision.models import resnet18

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 1000, bias=True),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1000, 4, bias=True))
model.to(CFG.device)
optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr, weight_decay = 0.001)
num_train_steps = len(train_dataset) / CFG.batch_size * CFG.num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = len(train_dataset) / CFG.batch_size * 5, num_training_steps = num_train_steps)
loss_fn = torch.nn.CrossEntropyLoss()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1338675035.py in <cell line: 0>()
      7                                    nn.Dropout(p=0.5),
      8                                    nn.Linear(1000, 4, bias=True))
----> 9 model.to(CFG.device)
     10 optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr, weight_decay = 0.001)
     11 num_train_steps = len(train_dataset) / CFG.batch_size * CFG.num_epochs

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in to(self, *args, **kwargs)
   1341                     raise
   1342 
-> 1343         return self._apply(convert)
   1344 
   1345     def register_full_backward_pre_hook(

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    901         if recurse:
    902             for module in self.children():
--> 903                 module._apply(fn)
    904 
    905         def compute_should_use_set_data(tensor, tensor_applied):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    928             # `with torch.no_grad():`
    929             with torch.no_grad():
--> 930                 param_applied = fn(param)
    931             p_should_use_set_data = compute_should_use_set_data(param, param_applied)
    932 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in convert(t)
   1327                         memory_format=convert_to_format,
   1328                     )
-> 1329                 return t.to(
   1330                     device,
   1331                     dtype if t.is_floating_point() or t.is_complex() else None,

/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py in _lazy_init()
    317         if "CUDA_MODULE_LOADING" not in os.environ:
    318             os.environ["CUDA_MODULE_LOADING"] = "LAZY"
--> 319         torch._C._cuda_init()
    320         # Some of the queued calls may reentrantly call _lazy_init();
    321         # we need to just return without initializing in that case.

RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

## === cell 26
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

## === cell 27
len(train_loader)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006441096.py in <cell line: 0>()
----> 1 len(train_loader)

NameError: name 'train_loader' is not defined

## === cell 31
best_valid_loss = float('inf')
patience = 3
trigger_times = 0

for epoch in range(8):
    tl, ta = train_fn(model, loader=train_loader)
    vl, va, conf_matrix = valid_fn(model, loader=valid_loader)

    if vl < best_valid_loss:
        best_valid_loss = vl
        torch.save(model.state_dict(), f'best_model.pt')
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1174148075.py in <cell line: 0>()
      4 
      5 for epoch in range(8):
----> 6     tl, ta = train_fn(model, loader=train_loader)
      7     vl, va, conf_matrix = valid_fn(model, loader=valid_loader)
      8 

NameError: name 'train_loader' is not defined

## === cell 32
plt.figure()
plt.ylim(0, 1.5)
sns.lineplot(x=list(range(len(train_loss))), y=train_loss, label='Train Loss')
sns.lineplot(x=list(range(len(valid_loss))), y=valid_loss, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

## === cell 33
subs = []
out = test_fn(model, test_loader)
output = pd.DataFrame(softmax(out,1), columns = ['healthy','multiple_diseases','rust','scab']) 
output.drop(0, inplace = True)
output.reset_index(drop=True,inplace=True)
subs.append(output)
sub_eff2 = sum(subs)/5

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2858284056.py in <cell line: 0>()
      1 subs = []
----> 2 out = test_fn(model, test_loader)
      3 output = pd.DataFrame(softmax(out,1), columns = ['healthy','multiple_diseases','rust','scab'])
      4 output.drop(0, inplace = True)
      5 output.reset_index(drop=True,inplace=True)

NameError: name 'test_loader' is not defined

## === cell 34
sub2 = sub_eff2.copy()
sub2['image_id'] = test.image_id
sub2 = sub2[['image_id','healthy','multiple_diseases','rust','scab']]
sub2.to_csv('submission_efficientnet2.csv', index = False)

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3825307639.py in <cell line: 0>()
----> 1 sub2 = sub_eff2.copy()
      2 sub2['image_id'] = test.image_id
      3 sub2 = sub2[['image_id','healthy','multiple_diseases','rust','scab']]
      4 sub2.to_csv('submission_efficientnet2.csv', index = False)

NameError: name 'sub_eff2' is not defined

## === cell 35
sub = sub_eff2
sub['image_id'] = test.image_id
sub = sub[['image_id','healthy','multiple_diseases','rust','scab']]
sub.to_csv('submission.csv', index = False)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1069309947.py in <cell line: 0>()
----> 1 sub = sub_eff2
      2 sub['image_id'] = test.image_id
      3 sub = sub[['image_id','healthy','multiple_diseases','rust','scab']]
      4 sub.to_csv('submission.csv', index = False)

NameError: name 'sub_eff2' is not defined

## === cell 37
import torch.onnx

def Convert_to_ONNX():
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "Leaf_Classifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

Convert_to_ONNX()

## === cell 38
import onnxruntime as ort

def preprocess_of_image(img):
    
        
        
        transforming = 
        
        
        if self.test is False:
            label = torch.tensor(np.argmax(self.labels.loc[item].values))
        if self.train is True:
            img = self.train_transform(image=img)['image']
            img = self.default_transform(image=img)['image']
        
        
        self.test_transform = Compose([
            HorizontalFlip(p = 0.5),
            VerticalFlip(p = 0.5),
            ShiftScaleRotate(rotate_limit=25.0, p=0.7),
            Resize(height=224, width=224)
        ])
        self.default_transform = Compose([
            Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                always_apply=True),
                ToTensorV2()
        ])

## --- ERROR in cell 38, traceback:
  File "/tmp/ipykernel_11/316432208.py", line 9
    transforming =
                   ^
SyntaxError: invalid syntax
