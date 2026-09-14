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

3.12

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
timm==1.0.19
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

0.0780772011140781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torchvision import transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score

import copy
import timm

## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augmentations=None, max_count=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_count = max_count
        self.test = test
        
        if not test:
            self.class_counts = self.annotations['diagnosis'].value_counts().sort_index()
        else:
            self.class_counts = None
        
        if max_count:
            self.oversample(max_count)
    
    def oversample(self, max_count): ## Over sampling classes to balance
        samples = []
        for diagnosis in self.class_counts.index:
            class_samples = self.annotations[self.annotations['diagnosis'] == diagnosis]
            oversampled_class = class_samples.sample(max_count, replace=True)
            samples.append(oversampled_class)
        self.annotations = pd.concat(samples).reset_index(drop=True)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.test:
            return image
                
        label = int(self.annotations.iloc[idx, 1])
        return image, label

## === cell 2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## === cell 3
def select_model(model_name, input_size):
    if model_name == 'efficientnet_b5':
        return timm.create_model('efficientnet_b5', pretrained=False, num_classes=5, in_chans=3)
    elif model_name == 'inception_resnet_v2':
        return timm.create_model('inception_resnet_v2', pretrained=False, num_classes=5, in_chans=3)
    elif model_name == 'inception_v4':
        return timm.create_model('inception_v4', pretrained=False, num_classes=5, in_chans=3)
    elif model_name == 'seresnext50_32x4d':
        return timm.create_model('seresnext50_32x4d', pretrained=False, num_classes=5, in_chans=3)
    elif model_name == 'seresnext101_32x4d':
        return timm.create_model('seresnext101_32x4d', pretrained=False, num_classes=5, in_chans=3)
    elif model_name == "resnet18(WD_1e-3)_aptos":
        return timm.create_model('resnet18', pretrained=False, num_classes=5, in_chans=3)
    else:
        raise ValueError(f"Unknown model name {model_name}")

## === cell 4
def load_model(model_name, input_size, model_path, device):
    model = select_model(model_name, input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

## === cell 5
def load_all_models(model_paths, device):
    models = []
    for model_name, input_size, model_path in model_paths:
        model = load_model(model_name, input_size, model_path, device)
        models.append((model_name, model))
    return models

## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1"
os.makedirs(save_dir, exist_ok=True)

## === cell 7
"""
model_paths = [
    ('efficientnet_b5', 512, os.path.join(save_dir, 'efficientnet_b5.pth')),
    ('resnet18(WD_1e-3)_aptos', 512, os.path.join(save_dir, 'resnet18(WD_1e-3)_aptos.pth')),
    ('inception_resnet_v2', 512, os.path.join(save_dir, 'inception_resnet_v2.pth')),
    ('inception_v4', 512, os.path.join(save_dir, 'inception_v4.pth')),
    ('seresnext50_32x4d', 512, os.path.join(save_dir, 'seresnext50_32x4d.pth')),
    ('seresnext101_32x4d', 384, os.path.join(save_dir, 'seresnext101_32x4d.pth'))
]
"""

model_paths = [
    ('resnet18(WD_1e-3)_aptos', 512, os.path.join(save_dir, 'resnet18(WD_1e-3)_aptos.pth'))
]

## === cell 8
loaded_models = load_all_models(model_paths, device)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3950845325.py in <cell line: 0>()
----> 1 loaded_models = load_all_models(model_paths, device)

/tmp/ipykernel_11/1552105129.py in load_all_models(model_paths, device)
      2     models = []
      3     for model_name, input_size, model_path in model_paths:
----> 4         model = load_model(model_name, input_size, model_path, device)
      5         models.append((model_name, model))
      6     return models

/tmp/ipykernel_11/4097719948.py in load_model(model_name, input_size, model_path, device)
      1 def load_model(model_name, input_size, model_path, device):
      2     model = select_model(model_name, input_size).to(device)
----> 3     model.load_state_dict(torch.load(model_path))
      4     model.eval()
      5     return model

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth'

## === cell 9
def predict_ensemble(models, test_csv_file, test_root_dir, submission_file, device):
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    test_dataset = BlindnessDataset(test_csv_file, test_root_dir, transform=transform, test=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    predictions = []
    for model_name, model in models:
        model_preds = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    preds = torch.softmax(outputs, dim=1)
                model_preds.append(preds.cpu().numpy())
        model_preds = np.concatenate(model_preds, axis=0)
        predictions.append(model_preds)

    avg_predictions = np.mean(predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)

    test_df = pd.read_csv(test_csv_file)
    submission_df = pd.DataFrame({'id_code': test_df['id_code'], 'diagnosis': final_predictions})
    submission_df.to_csv(submission_file, index=False)

## === cell 10
test_csv_file = '/kaggle/input/aptos2019-blindness-detection/test.csv'
test_root_dir = '/kaggle/input/aptos2019-blindness-detection/test_images'
submission_file = '/kaggle/working/submission.csv'
predict_ensemble(loaded_models, test_csv_file, test_root_dir, submission_file, device)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2985076849.py in <cell line: 0>()
      2 test_root_dir = '/kaggle/input/aptos2019-blindness-detection/test_images'
      3 submission_file = '/kaggle/working/submission.csv'
----> 4 predict_ensemble(loaded_models, test_csv_file, test_root_dir, submission_file, device)

NameError: name 'loaded_models' is not defined
