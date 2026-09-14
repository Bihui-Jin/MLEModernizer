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

0.0383495739528566

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
import os
import cv2
import sys
import torch
import dicomsdl
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from sklearn.model_selection import train_test_split

sys.path.append('/kaggle/input/library-load/efficientnet_pytorch-0.7.1') 
from efficientnet_pytorch import EfficientNet

%matplotlib inline

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4068721465.py in <cell line: 0>()
      3 import sys
      4 import torch
----> 5 import dicomsdl
      6 import numpy as np
      7 import pandas as pd

ModuleNotFoundError: No module named 'dicomsdl'

## === cell 5
data_dir = '/kaggle/input/rsna-breast-cancer-detection'
tmp_output_dir = '/kaggle/tmp/output'

train_dir = os.path.join(tmp_output_dir, 'train_images')
test_dir = os.path.join(tmp_output_dir, 'test_images')

## === cell 6
target_size = [224,224] 
batch_size = 16 
num_epochs = 6 

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
----> 1 train_df = pd.read_csv(f"{data_dir}/train.csv")
      2 
      3 train_df['dcm_path'] = train_df.apply(
      4     lambda i: os.path.join(
      5         f"{data_dir}", 'train_images', str(i['patient_id']), str(i['image_id']) + '.dcm'

NameError: name 'pd' is not defined

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
----> 1 test_df = pd.read_csv(f"{data_dir}/test.csv")
      2 
      3 test_df['dcm_path'] = test_df.apply(
      4     lambda i: os.path.join(
      5         f"{data_dir}", 'test_images', str(i['patient_id']), str(i['image_id']) + '.dcm'

NameError: name 'pd' is not defined

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
/tmp/ipykernel_11/4021771097.py in <cell line: 0>()
----> 1 class MyDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.transform = transform
      5 

NameError: name 'Dataset' is not defined

## === cell 42
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1, weights_path="/kaggle/input/model-files/efficientnet-b0-355c32eb.pth")
        
    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3326557632.py in <cell line: 0>()
----> 1 class PretrainedBinaryClassifier(nn.Module):
      2     def __init__(self):
      3         super(PretrainedBinaryClassifier, self).__init__()
      4         # Load the pre-trained EfficientNet model
      5         self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1, weights_path="/kaggle/input/model-files/efficientnet-b0-355c32eb.pth")

NameError: name 'nn' is not defined

## === cell 46
random_seed = 534
train_subset_0 = train_df[train_df.cancer == 0].sample(n=20, random_state=random_seed)
train_subset_1 = train_df[train_df.cancer == 1].sample(n=10, random_state=random_seed) 
train_combined = pd.concat([train_subset_0, train_subset_1])

print(train_combined.shape)
print(train_combined.laterality.value_counts())
print(train_combined.cancer.value_counts())
train_combined.reset_index(inplace=True)

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2113790684.py in <cell line: 0>()
      1 # set seed for reproducibility
      2 random_seed = 534
----> 3 train_subset_0 = train_df[train_df.cancer == 0].sample(n=20, random_state=random_seed)
      4 train_subset_1 = train_df[train_df.cancer == 1].sample(n=10, random_state=random_seed)
      5 train_combined = pd.concat([train_subset_0, train_subset_1])

NameError: name 'train_df' is not defined

## === cell 47
training_set, validation_set = train_test_split(train_combined, test_size=0.2, random_state=276)
print(training_set.shape)
print(validation_set.shape)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2772988165.py in <cell line: 0>()
----> 1 training_set, validation_set = train_test_split(train_combined, test_size=0.2, random_state=276)
      2 print(training_set.shape)
      3 print(validation_set.shape)

NameError: name 'train_test_split' is not defined

## === cell 49
%%time



class ToTensorWithErasing(object):
    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        img_tensor = transforms.RandomErasing(p=0.5, scale=(0.1, 0.5), ratio=(0.3, 3.3))(img_tensor)
        return img_tensor

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.7, 1.3)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
    ToTensorWithErasing(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([transforms.ToTensor()]) #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = MyDataset(training_set, transform = train_transform) 
val_dataset = MyDataset(validation_set, transform = val_transform) 

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'transforms' is not defined

## === cell 51
if len(train_dataset) > 0:
    print("The training dataset contains", len(train_dataset), "samples.")
else:
    print("The training dataset is empty.")
    
if len(val_dataset) > 0:
    print("The val dataset contains", len(val_dataset), "samples.")
else:
    print("The val dataset is empty.")

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3262194588.py in <cell line: 0>()
----> 1 if len(train_dataset) > 0:
      2     print("The training dataset contains", len(train_dataset), "samples.")
      3 else:
      4     print("The training dataset is empty.")
      5 

NameError: name 'train_dataset' is not defined

## === cell 52
print("checking if transformation is applied")
print("Training dataset transform is: ", train_dataset.transform)
print("Validation dataset transform is:", val_dataset.transform)

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1540639282.py in <cell line: 0>()
      1 print("checking if transformation is applied")
----> 2 print("Training dataset transform is: ", train_dataset.transform)
      3 print("Validation dataset transform is:", val_dataset.transform)

NameError: name 'train_dataset' is not defined

## === cell 54
%%time
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 1)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'DataLoader' is not defined

## === cell 55
print("Number of training batches",len(train_loader)) 
print("Number of validation batches",len(val_loader))

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/228137181.py in <cell line: 0>()
      2 # so this 4.125 batches are passed through each epoch. The weights are updated after each epoch and the model continues to learn. Usually, we try to stop the model learning if
      3 #no changes in loss or accuracy takes place. We do this using the concept of early stopping.
----> 4 print("Number of training batches",len(train_loader))
      5 print("Number of validation batches",len(val_loader))

NameError: name 'train_loader' is not defined

## === cell 57
print("Number of training examples before augmentation:", len(training_set))
print("Number of training examples after augmentation:", len(train_dataset))

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3975580003.py in <cell line: 0>()
----> 1 print("Number of training examples before augmentation:", len(training_set))
      2 print("Number of training examples after augmentation:", len(train_dataset))

NameError: name 'training_set' is not defined

## === cell 61
print(torch.cuda.is_available())
print(torch.cuda.device_count()) # print the number of available GPUs on the current system. If no GPUs are available, it would print 0.

## === cell 62
model = PretrainedBinaryClassifier()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss() 
criterion = criterion.cuda()

learning_rate = 0.0001
weight_decay = 0.01

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2178476055.py in <cell line: 0>()
      1 # model = BinaryClassifier()
----> 2 model = PretrainedBinaryClassifier()
      3 
      4 # Move the model to the GPU device
      5 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NameError: name 'PretrainedBinaryClassifier' is not defined

## === cell 63
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

## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'model' is not defined

## === cell 64
print(len(train_losses)) 
print(len(train_acc_metric))
print(len(train_pf1_metric))
print(train_losses)
print(train_acc_metric)
print(train_pf1_metric)

## === cell 65
print(len(val_losses))
print(len(val_acc_metric))
print(len(val_pf1_metric))
print(val_losses)
print(val_acc_metric)
print(val_pf1_metric)

## === cell 66
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

## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3476470521.py in <cell line: 0>()
      1 # Plot the learning curves
----> 2 plt.plot(train_losses, label='Training Loss')
      3 plt.plot(val_losses, label='Validation Loss')
      4 plt.xlabel('Epoch')
      5 plt.ylabel('Loss')

NameError: name 'plt' is not defined

## === cell 70
test_df.head()

## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1122697411.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 72
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


## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3677148535.py in <cell line: 0>()
----> 1 class TestDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.transform = transform
      5 

NameError: name 'Dataset' is not defined

## === cell 73
%%time

test_batch_size = 32 

test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = TestDataset(test_df, transform = test_transform)
test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers = 1)



## --- ERROR in cell 73, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'transforms' is not defined

## === cell 74
%%time

predictions = []

for batch in test_dataloader:
    pids, images = batch
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        predictions.extend(probabilities.cpu().numpy())

## --- ERROR in cell 74, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'test_dataloader' is not defined

## === cell 75
def get_val(x):
    return x[0]

test_df['cancer'] = predictions
test_df['cancer'] = test_df['cancer'].apply(lambda x: get_val(x))
test_df.tail()



## --- ERROR in cell 75, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/173606790.py in <cell line: 0>()
      2     return x[0]
      3 
----> 4 test_df['cancer'] = predictions
      5 test_df['cancer'] = test_df['cancer'].apply(lambda x: get_val(x))
      6 test_df.tail()

NameError: name 'test_df' is not defined

## === cell 76
sub = test_df.groupby('prediction_id')['cancer'].mean().to_frame().reset_index()
sub.tail()


## --- ERROR in cell 76, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1197230213.py in <cell line: 0>()
----> 1 sub = test_df.groupby('prediction_id')['cancer'].mean().to_frame().reset_index()
      2 sub.tail()
      3 # sub = new_test_df.groupby('prediction_id')['cancer'].mean().to_frame().reset_index()
      4 # sub.tail()

NameError: name 'test_df' is not defined

## === cell 77
sub.to_csv("submission.csv", index = False)

## --- ERROR in cell 77, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/745902861.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index = False)

NameError: name 'sub' is not defined

## === cell 79
# """
# This code cell is to check if the augmentations get applied correcty or not.

# Note: The images present here in the train_loader should be unaugmented one.
