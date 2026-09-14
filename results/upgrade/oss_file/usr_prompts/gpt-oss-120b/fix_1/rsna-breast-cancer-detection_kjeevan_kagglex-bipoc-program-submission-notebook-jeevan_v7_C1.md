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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.0381862601745652

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
pip install /kaggle/input/my-requirement-files/dicomsdl-0.109.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

## === cell 3
import numpy as np
import pandas as pd
import seaborn as sns
import dicomsdl
import cv2
import os
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
%matplotlib inline

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3004323558.py in <cell line: 0>()
      2 import pandas as pd
      3 import seaborn as sns
----> 4 import dicomsdl
      5 import cv2
      6 import os

ModuleNotFoundError: No module named 'dicomsdl'

## === cell 5
data_dir = '/kaggle/input/rsna-breast-cancer-detection'
tmp_output_dir = '/kaggle/tmp/output'

train_dir = os.path.join(tmp_output_dir, 'train_images')
test_dir = os.path.join(tmp_output_dir, 'test_images')

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3143001290.py in <cell line: 0>()
      2 tmp_output_dir = '/kaggle/tmp/output'
      3 
----> 4 train_dir = os.path.join(tmp_output_dir, 'train_images')
      5 test_dir = os.path.join(tmp_output_dir, 'test_images')

NameError: name 'os' is not defined

## === cell 6
target_size = [216,216] #[512, 512]
batch_size = 64 #128 #64 
num_epochs = 5 #2

## === cell 7
train_df = pd.read_csv(f"{data_dir}/train.csv")

train_df['dcm_path'] = train_df.apply(
    lambda i: os.path.join(
        f"{data_dir}", 'train_images', str(i['patient_id']), str(i['image_id']) + '.dcm'
    ), axis=1
)
print(train_df.shape)
train_df.head(2)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/143854144.py in <cell line: 0>()
      1 train_df = pd.read_csv(f"{data_dir}/train.csv")
      2 
----> 3 train_df['dcm_path'] = train_df.apply(
      4     lambda i: os.path.join(
      5         f"{data_dir}", 'train_images', str(i['patient_id']), str(i['image_id']) + '.dcm'

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/143854144.py in <lambda>(i)
      2 
      3 train_df['dcm_path'] = train_df.apply(
----> 4     lambda i: os.path.join(
      5         f"{data_dir}", 'train_images', str(i['patient_id']), str(i['image_id']) + '.dcm'
      6     ), axis=1

NameError: name 'os' is not defined

## === cell 8
test_df = pd.read_csv(f"{data_dir}/test.csv")

test_df['dcm_path'] = test_df.apply(
    lambda i: os.path.join(
        f"{data_dir}", 'test_images', str(i['patient_id']), str(i['image_id']) + '.dcm'
    ), axis=1
)
print(test_df.shape)
test_df.head(2)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2834979546.py in <cell line: 0>()
      1 test_df = pd.read_csv(f"{data_dir}/test.csv")
      2 
----> 3 test_df['dcm_path'] = test_df.apply(
      4     lambda i: os.path.join(
      5         f"{data_dir}", 'test_images', str(i['patient_id']), str(i['image_id']) + '.dcm'

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/2834979546.py in <lambda>(i)
      2 
      3 test_df['dcm_path'] = test_df.apply(
----> 4     lambda i: os.path.join(
      5         f"{data_dir}", 'test_images', str(i['patient_id']), str(i['image_id']) + '.dcm'
      6     ), axis=1

NameError: name 'os' is not defined

## === cell 39
def normalize_xray(path, fix_monochrome = True):
    dicom = dicomsdl.open(path)
    dicom_arr = dicom.pixelData(storedvalue=False)  # storedvalue = True for int16 return otherwise float32    
    dicom_arr = dicom_arr - np.min(dicom_arr)    
    dicom_arr = dicom_arr / np.max(dicom_arr)    
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        dicom_arr = np.amax(dicom_arr) - dicom_arr    
    dicom_arr = dicom_arr - np.min(dicom_arr)
    dicom_arr = dicom_arr / np.max(dicom_arr)
    dicom_arr = (dicom_arr * 255).astype(np.uint8)
    return dicom_arr

def crop_and_resize(image, crop_size=5):
    image = image[crop_size:-crop_size, crop_size:-crop_size]
    image = cv2.resize(image, target_size[::-1], cv2.INTER_LINEAR)
    return image

def preprocess_and_save(file_path):
    image = normalize_xray(file_path)
    h, w = image.shape[:2]  # orig hw, so we take the height and width
    image = crop_and_resize(image)
    sub_path = file_path.split("/",4)[-1].split('.dcm')[0] + '.png'
    infos = sub_path.split('/')
    pid = infos[-2]
    iid = infos[-1]; iid = iid.replace('.png','')
    return pid, iid, h, w, image

def pfbeta_torch(labels, preds, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0

## === cell 40
class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        cancer = row['cancer']
        cancer = torch.tensor(cancer, dtype=torch.long) 
        path = row['dcm_path']
        path_parts = path.split('/')
        filename = path_parts[-1]
        full_path = os.path.join('/'.join(path_parts[:-1]), filename)
        pid, iid, h, w, img = preprocess_and_save(full_path)
        img = Image.fromarray(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'cancer': cancer, 'images': img}  

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2992339058.py in <cell line: 0>()
----> 1 class MyDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.transform = transform
      5 

NameError: name 'Dataset' is not defined

## === cell 42
import torchvision.models as models
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)
        state_dict = torch.load('/kaggle/input/my-requirement-files/resnet50-0676ba61.pth')
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.sigmoid(x)
        return x

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3123528413.py in <cell line: 0>()
      1 import torchvision.models as models
----> 2 class PretrainedBinaryClassifier(nn.Module):
      3     def __init__(self):
      4         super(PretrainedBinaryClassifier, self).__init__()
      5         # Load a pre-trained model (ResNet50)

NameError: name 'nn' is not defined

## === cell 45
train_subset_0 = train_df[train_df.cancer == 0].iloc[:55,:] #400, 3000, 430
train_subset_1 = train_df[train_df.cancer == 1].iloc[:45,:] #200, 1158, 350
train_combined = pd.concat([train_subset_0, train_subset_1])
print(train_combined.shape)
print(train_combined.laterality.value_counts())
print(train_combined.cancer.value_counts())
train_combined.reset_index(inplace=True)

## === cell 46
training_set, validation_set = train_test_split(train_combined, test_size=0.2, random_state=42)
print(training_set.shape)
print(validation_set.shape)

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/571812803.py in <cell line: 0>()
----> 1 training_set, validation_set = train_test_split(train_combined, test_size=0.2, random_state=42)
      2 print(training_set.shape)
      3 print(validation_set.shape)

NameError: name 'train_test_split' is not defined

## === cell 48
%%time

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MyDataset(training_set, transform=transform)
val_dataset = MyDataset(validation_set, transform=transform)

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'transforms' is not defined

## === cell 49
if len(train_dataset) > 0:
    print("The training dataset contains", len(train_dataset), "samples.")
else:
    print("The training dataset is empty.")
    
if len(val_dataset) > 0:
    print("The val dataset contains", len(val_dataset), "samples.")
else:
    print("The val dataset is empty.")

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3262194588.py in <cell line: 0>()
----> 1 if len(train_dataset) > 0:
      2     print("The training dataset contains", len(train_dataset), "samples.")
      3 else:
      4     print("The training dataset is empty.")
      5 

NameError: name 'train_dataset' is not defined

## === cell 51
%%time
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'DataLoader' is not defined

## === cell 52
print(len(train_loader)) 
print(len(val_loader))

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2166593204.py in <cell line: 0>()
      2 # so this 4.125 batches are passed through each epoch. The weights are updated after each epoch and the model continues to learn. Usually, we try to stop the model learning if
      3 #no changes in loss or accuracy takes place. We do this using the concept of early stopping.
----> 4 print(len(train_loader))
      5 print(len(val_loader))

NameError: name 'train_loader' is not defined

## === cell 55
print(torch.cuda.is_available())
print(torch.cuda.device_count())

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3841601903.py in <cell line: 0>()
----> 1 print(torch.cuda.is_available())
      2 print(torch.cuda.device_count())

NameError: name 'torch' is not defined

## === cell 56
model = PretrainedBinaryClassifier()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.BCEWithLogitsLoss() 
criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001) # 0.001


## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1861779567.py in <cell line: 0>()
      1 # model = BinaryClassifier()
----> 2 model = PretrainedBinaryClassifier()
      3 
      4 # Move the model to the GPU device
      5 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NameError: name 'PretrainedBinaryClassifier' is not defined

## === cell 57
%%time

train_losses = []
train_acc_metric = []
train_pf1_metric = []

val_losses = []
val_acc_metric = []
val_pf1_metric = []

for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_train_acc = 0.0
    running_train_pf1 = 0
    
    model.train() # set the model to training mode

    for i, data in enumerate(train_loader, 0):
        inputs, targets = data['images'], data['cancer']
        targets = targets.view(-1, 1) # Reshape the target tensor to (batch_size, 1)
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        
        predicted = torch.round(torch.sigmoid(outputs))
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
        pf1_metric = pfbeta_torch(targets.cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += accuracy
        running_train_pf1 += pf1_metric


    avg_train_loss = running_train_loss / len(train_loader)
    avg_train_acc = running_train_acc / len(train_loader)
    avg_train_pf1 = running_train_pf1 / len(train_loader)
    train_losses.append(avg_train_loss)
    train_acc_metric.append(avg_train_acc) # train_metrics
    train_pf1_metric.append(avg_train_pf1)
    print('Epoch %d, avg training loss: %.3f, avg training accuracy: %.3f, avg training pf1: %.3f'%
          (epoch + 1, avg_train_loss, avg_train_acc, avg_train_pf1))


    model.eval() # set the model to evaluation mode
    running_val_loss = 0.0
    running_val_acc = 0.0
    running_val_pf1 = 0

    with torch.no_grad():

        for i, data in enumerate(val_loader, 0):
            inputs, targets = data['images'], data['cancer']
            targets = targets.view(-1, 1) # Reshape the target tensor to (batch_size, 1)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets.float())

            predicted = torch.round(torch.sigmoid(outputs))
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)
            
            pf1_metric = pfbeta_torch(targets.cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())

            running_val_loss += loss.item()
            running_val_acc += accuracy
            running_val_pf1 += pf1_metric

    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_acc = running_val_acc / len(val_loader)
    avg_val_pf1 = running_val_pf1 / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc_metric.append(avg_val_acc)
    val_pf1_metric.append(avg_val_pf1)
    print('Epoch %d, avg validation loss: %.3f, avg validation accuracy: %.3f, avg validation pf1: %.3f' %
          (epoch + 1, avg_val_loss, avg_val_acc, avg_val_pf1))

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'model' is not defined

## === cell 59
print(len(train_losses))
print(len(train_acc_metric))
print(len(train_pf1_metric))
print(train_losses)
print(train_acc_metric)
print(train_pf1_metric)

## === cell 60
print(len(val_losses))
print(len(val_acc_metric))
print(len(val_pf1_metric))
print(val_losses)
print(val_acc_metric)
print(val_pf1_metric)

## === cell 61
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(train_acc_metric, label='Training Accuracy')
plt.plot(val_acc_metric, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(train_pf1_metric, label='Training Probabilistic F1 Score')
plt.plot(val_pf1_metric, label='Validation Probabilistic F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Probabilistic F1 Score')
plt.legend()
plt.show()

## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3476470521.py in <cell line: 0>()
      1 # Plot the learning curves
----> 2 plt.plot(train_losses, label='Training Loss')
      3 plt.plot(val_losses, label='Validation Loss')
      4 plt.xlabel('Epoch')
      5 plt.ylabel('Loss')

NameError: name 'plt' is not defined

## === cell 65
test_df.head()

## === cell 67
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['dcm_path']
        path_parts = path.split('/')
        patient_id = path_parts[-2]
        filename = path_parts[-1]
        full_path = os.path.join('/'.join(path_parts[:-1]), filename)
        pid, iid, h, w, img = preprocess_and_save(full_path)
        img = Image.fromarray(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return patient_id, img

## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1203785533.py in <cell line: 0>()
----> 1 class TestDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.transform = transform
      5 

NameError: name 'Dataset' is not defined

## === cell 68
512*2

## === cell 69
%%time

batch_size = 128 #64 # Set a batch size that fits in your GPU memory
num_workers = 2 # Set the number of CPU workers to use for loading the data

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = TestDataset(test_df, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
len(test_dataset)


## --- ERROR in cell 69, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'transforms' is not defined

## === cell 70
len(test_dataloader) # if batch size is 64 then number of batches will be 126

## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/97220999.py in <cell line: 0>()
----> 1 len(test_dataloader) # if batch size is 64 then number of batches will be 126

NameError: name 'test_dataloader' is not defined

## === cell 71
%%time

predictions = []

for batch in test_dataloader:
    pids, images = batch
    
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        predictions.extend(probabilities.cpu().numpy())


## --- ERROR in cell 71, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'test_dataloader' is not defined

## === cell 72
pd.DataFrame(predictions).tail() #nbjh

## === cell 73
def get_val(x):
    return x[0]

test_df['cancer'] = predictions
test_df['cancer'] = test_df['cancer'].apply(lambda x: get_val(x))
test_df.tail()



## --- ERROR in cell 73, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/173606790.py in <cell line: 0>()
      2     return x[0]
      3 
----> 4 test_df['cancer'] = predictions
      5 test_df['cancer'] = test_df['cancer'].apply(lambda x: get_val(x))
      6 test_df.tail()

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

ValueError: Length of values (0) does not match length of index (5474)

## === cell 74
sub = test_df.groupby('prediction_id')['cancer'].mean().to_frame().reset_index()
sub.tail()


## --- ERROR in cell 74, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1197230213.py in <cell line: 0>()
----> 1 sub = test_df.groupby('prediction_id')['cancer'].mean().to_frame().reset_index()
      2 sub.tail()
      3 # sub = new_test_df.groupby('prediction_id')['cancer'].mean().to_frame().reset_index()
      4 # sub.tail()

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in __getitem__(self, key)
   1949                 "Use a list instead."
   1950             )
-> 1951         return super().__getitem__(key)
   1952 
   1953     def _gotitem(self, key, ndim: int, subset=None):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in __getitem__(self, key)
    242         else:
    243             if key not in self.obj:
--> 244                 raise KeyError(f"Column not found: {key}")
    245             ndim = self.obj[key].ndim
    246             return self._gotitem(key, ndim=ndim)

KeyError: 'Column not found: cancer'

## === cell 75
sub.to_csv("submission.csv", index = False)

## --- ERROR in cell 75, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/745902861.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index = False)

NameError: name 'sub' is not defined
