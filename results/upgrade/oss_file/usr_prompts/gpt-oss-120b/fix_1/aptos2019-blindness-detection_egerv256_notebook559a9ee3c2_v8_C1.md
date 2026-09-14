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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.9

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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.0542674972869088

# 6. Current score

0.02895

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm_notebook
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.utils.data as utils
import torchvision.models as models

import torchvision
from torchvision import transforms

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")
IMG_SIZE = 224

## === cell 1
pretrained_model = torchvision.models.densenet161(pretrained=False, num_classes=5)

## === cell 2
data_dir = '../input/aptos2019-blindness-detection/'
train_dir = data_dir + '/train_images/'
test_dir = data_dir + '/test_images/'

## === cell 3
df_train = pd.read_csv(data_dir+"train.csv")
df_test = pd.read_csv(data_dir+"test.csv")
df_test['diagnosis'] = -1

## === cell 4
df_train.head()

## === cell 5
df_test.head()

## === cell 6
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

## === cell 7
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

## === cell 8
class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_name,label = self.df[index]
        
        img_path = os.path.join(self.data_dir, img_name+'.png')
        image = load_ben_color(img_path,sigmaX=10)
        if self.transform:
            image = self.transform(image)
        return image, label

## === cell 9
data_transf = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomRotation((-180, 180)),
    torchvision.transforms.RandomHorizontalFlip(p=0.4),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

## === cell 10
def train_one_fold(i_fold, model, loss_func, optimizer, train_loader, valid_loader, n_epochs):
    best_loss = float('+inf')
    train_fold_results = []

    for epoch in range(n_epochs):
        train_start = time.time()

        print('  Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('  ' + ('-' * 20))

        model.train()
        tr_loss = 0
        train_kappa = []
        optimizer.zero_grad()
        for ii, (data, target) in tqdm_notebook(enumerate(train_loader), total=len(train_loader)):

            images = data.to('cuda', dtype = torch.float)  
            labels = target.to('cuda', dtype = torch.float)  
            
            labels = labels.view(-1,1)
            optimizer.zero_grad()
            
            
            with torch.set_grad_enabled(True):
                output = model(images)   
                
                loss = loss_func(output, labels)
                loss.backward()

                optimizer.step()
            
            tr_loss += loss.item()
            
            y_actual = labels.data.cpu().numpy()
            y_pred = output[:,-1].detach().cpu().numpy()
            kappa = cohen_kappa_score(y_actual, y_pred.round(), weights='quadratic')
            train_kappa.append(kappa)
        
        train_kappa_epoch = np.mean(train_kappa)

        train_end = time.time()
            
        val_start = time.time()
        model.eval()
        
        val_loss = 0
        val_preds = None
        val_labels = None
        val_kappa = []
        
        for ii, (data, target) in tqdm_notebook(enumerate(valid_loader), total=len(valid_loader)):


            images = data.to('cuda', dtype = torch.float)
            labels = target.to('cuda', dtype = torch.float)
            labels = labels.view(-1,1)

            with torch.no_grad():
                outputs = model(images)
                
                loss = loss_func(outputs, labels)
                val_loss += loss.item()

            y_actual = labels.data.cpu().numpy()
            y_pred = outputs[:,-1].detach().cpu().numpy()
            kappa = cohen_kappa_score(y_actual, y_pred.round(), weights='quadratic')
            val_kappa.append(kappa)

           
        val_kappa_epoch = np.mean(val_kappa)
        val_end = time.time()
        train_loss = tr_loss/len(train_loader)
        valid_loss = val_loss/len(valid_loader)
        
        print('Fold: {}, Epoch: {}, Train duration: {:.6f}, Valid duration: {:.6f}, Train Loss: {:.6f}, Valid Loss: {:.6f}, Train Kappa: {:.4f}, Valid Kappa: {:.4f}'.format(i_fold+1, epoch+1, train_end - train_start, val_end - val_start, train_loss, valid_loss, train_kappa_epoch, val_kappa_epoch))
        if best_loss > valid_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), '../input/aptos-checkpoints/aptos_3_pretrained_model_161_{}_fold_{}_epoch.pth'.format(i_fold, epoch))
    return train_loss, valid_loss, train_kappa_epoch, val_kappa_epoch

## === cell 11
N_FOLDS = 2
BATCH_SIZE = 32
N_EPOCHS = 15

folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

## === cell 12
state_dict = torch.load('../input/aptos-checkpoints/aptos_3_pretrained_model_161_1_fold_0_epoch.pth', map_location=lambda storage, loc: storage)
pretrained_model.load_state_dict(state_dict)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1091608156.py in <cell line: 0>()
----> 1 state_dict = torch.load('../input/aptos-checkpoints/aptos_3_pretrained_model_161_1_fold_0_epoch.pth', map_location=lambda storage, loc: storage)
      2 pretrained_model.load_state_dict(state_dict)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/aptos-checkpoints/aptos_3_pretrained_model_161_1_fold_0_epoch.pth'

## === cell 13
def predict(model, testloader):
    '''Function used to make predictions on the test set'''
    model.eval()
    preds = []
    for batch_i, (data, target) in tqdm_notebook(enumerate(testloader), total=len(testloader)):
        data, target = data, target
        output = model(data)
        pr = output.argmax(dim=1).detach().cpu().numpy()
        for item in pr:
            preds.append(item.item())
            
    return preds

## === cell 14
test_data = ImageData(df = df_test, data_dir = test_dir, transform = transforms_valid)
test_loader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

## === cell 15
test_pred = predict(pretrained_model, test_loader)

## === cell 16
df_test['diagnosis'] = test_pred
df_test.to_csv('submission.csv',index=False)
df_test
