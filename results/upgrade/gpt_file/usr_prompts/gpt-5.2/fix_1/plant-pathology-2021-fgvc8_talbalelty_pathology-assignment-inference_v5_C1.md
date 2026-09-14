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

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.6083841181902121

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
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models

## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171) # image size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/train.csv')
test_images_path = '/kaggle/input/plant-pathology-2021-fgvc8/test_images/'
test_images_names = glob.glob(test_images_path + '*.jpg')

train_data['labels'] = train_data['labels'].apply(lambda string: string.split(' ')) # Create a list of labels
s = list(train_data['labels'])
mlb = MultiLabelBinarizer() 
train_labels = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index) # onehot encoding the labels in the list in 6 columns

labels_size = len(train_labels.columns)

## === cell 2
class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__() 
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):  
        if self.images_path is not None:
            image = cv2.imread(self.images_path+self.dataset.image[idx])
            labels = torch.tensor(self.dataset.loc[idx, self.dataset.columns != 'image'].tolist())
        else:
            image = cv2.imread(self.dataset[idx])
            labels = np.array([])
            
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, DIMENTION)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, labels

    def __len__(self):
        return len(self.dataset)

## === cell 3
transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                       ToTensorV2()])
test_dataset = PlantDataSet(test_images_names, None, transform)
BS = 30
plants_test_data_loader = DataLoader(dataset=test_dataset, batch_size=BS, shuffle=False)

## === cell 4
def test(test_dataloader, model):
    predictions = None
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(test_dataloader):
            images = images.float()    
            images = images.to(device)
            
            output = model(images)
            probabilities = torch.sigmoid(output)
            if i == 0:
                predictions = probabilities.detach().cpu().numpy()
            else:
                predictions = np.concatenate((predictions, probabilities.detach().cpu().numpy()), axis=0)

            del images
            torch.cuda.empty_cache()
            gc.collect()
            
    return np.array(predictions)

## === cell 5
def create_submission(test_images_path ,predictions):
    submission_df = pd.DataFrame(columns=['image', 'labels'])
    for image_name, prediction in zip(test_images_path, predictions):
        name = image_name.split('/')[-1]
        arr = [name for pred, name in zip(prediction, train_labels.columns) if pred > 0.75]
        if len(arr) == 0:
            arr = ['healthy']
            
        prediction_labels = " ".join(arr)

        row = pd.DataFrame([[name, prediction_labels]], columns=['image', 'labels'])
        submission_df = submission_df.append(row)
                

    submission_df = submission_df.reset_index(drop=True)
    display(submission_df)
    submission_df.to_csv('submission.csv', index=False)

## === cell 10
resnext50 = models.resnext50_32x4d(pretrained=False, num_classes=labels_size)
resnext50.load_state_dict(torch.load('../input/resnext50-32x4d-final/resnext50_32x4d_final.pth'))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3047424759.py in <cell line: 0>()
      1 # Load model
      2 resnext50 = models.resnext50_32x4d(pretrained=False, num_classes=labels_size)
----> 3 resnext50.load_state_dict(torch.load('../input/resnext50-32x4d-final/resnext50_32x4d_final.pth'))

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/resnext50-32x4d-final/resnext50_32x4d_final.pth'

## === cell 11
test_predictions = test(plants_test_data_loader, resnext50)
print(test_predictions)
create_submission(test_images_names, test_predictions)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/906891291.py in <cell line: 0>()
      1 test_predictions = test(plants_test_data_loader, resnext50)
      2 print(test_predictions)
----> 3 create_submission(test_images_names, test_predictions)

/tmp/ipykernel_11/1585471808.py in create_submission(test_images_path, predictions)
     10 
     11         row = pd.DataFrame([[name, prediction_labels]], columns=['image', 'labels'])
---> 12         submission_df = submission_df.append(row)
     13 
     14 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'
