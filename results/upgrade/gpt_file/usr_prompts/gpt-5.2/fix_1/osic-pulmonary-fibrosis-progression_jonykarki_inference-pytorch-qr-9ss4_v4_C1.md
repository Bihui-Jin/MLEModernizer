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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-7.361688716355977

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn import preprocessing

## === cell 1
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
MODEL_DIR = "/kaggle/input/osicqrmodel/"
QUANTILES = [0.2, 0.5, 0.8]
SCALE_COLUMNS = ['Weeks', 'FVC', 'Age']
SCALE_COLUMNS = ['Weeks_Passed', 'Base_FVC', 'Base_Percent', 'Base_Age']
SEX_COLUMNS = ['Male', 'Female']
SMOKING_STATUS_COLUMNS = ['Currently smokes', 'Ex-smoker', 'Never smoked']

FV = SEX_COLUMNS + SMOKING_STATUS_COLUMNS + SCALE_COLUMNS
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## === cell 2
MIN_MAX_SCALER = preprocessing.MinMaxScaler()

## === cell 3
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
train_df.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])

## === cell 4
sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[0])
sub_df['Weeks'] = sub_df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub_df.head()

## === cell 5
sub_df = sub_df.drop('FVC', axis=1).merge(test_df.drop('Weeks', axis=1), on='Patient')
sub_df.head()

## === cell 6
train_df['FROM'] = 'train'
test_df['FROM'] = 'val'
sub_df['FROM'] = 'test'

## === cell 7
combined_df = train_df.append([test_df, sub_df])

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3041607547.py in <cell line: 0>()
----> 1 combined_df = train_df.append([test_df, sub_df])

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 8
combined_df['Base_Week'] = combined_df['Weeks']
combined_df.loc[combined_df['FROM'] == 'test', 'Base_Week'] = np.nan
combined_df['Base_Week'] = combined_df.groupby('Patient')['Base_Week'].transform('min')

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3737101116.py in <cell line: 0>()
      1 # initialize base_week column
----> 2 combined_df['Base_Week'] = combined_df['Weeks']
      3 # make the weeks from sub_df to be np.nan so that when we calculate the base_week it comes from the test_df
      4 combined_df.loc[combined_df['FROM'] == 'test', 'Base_Week'] = np.nan
      5 # now calculate the min for each patient group and set it to the Base_Week column

NameError: name 'combined_df' is not defined

## === cell 9
base_df = combined_df[combined_df['Weeks'] == combined_df['Base_Week']]

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3545585569.py in <cell line: 0>()
      1 # get the base_df (where the Base_Week == the min_week we calculated) so that we can get the base_fvc, base_age and base_percentage
----> 2 base_df = combined_df[combined_df['Weeks'] == combined_df['Base_Week']]

NameError: name 'combined_df' is not defined

## === cell 10
base_df.rename(columns={
    'FVC': 'Base_FVC',
    'Percent': 'Base_Percent',
    'Age': 'Base_Age'
}, inplace=True)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4147407208.py in <cell line: 0>()
----> 1 base_df.rename(columns={
      2     'FVC': 'Base_FVC',
      3     'Percent': 'Base_Percent',
      4     'Age': 'Base_Age'
      5 }, inplace=True)

NameError: name 'base_df' is not defined

## === cell 11
combined_df = combined_df.merge(base_df[['Patient', 'Base_FVC', 'Base_Percent', 'Base_Age']], on='Patient', how='left')

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2299655535.py in <cell line: 0>()
----> 1 combined_df = combined_df.merge(base_df[['Patient', 'Base_FVC', 'Base_Percent', 'Base_Age']], on='Patient', how='left')

NameError: name 'combined_df' is not defined

## === cell 12
combined_df['Weeks_Passed'] = combined_df['Weeks'] - combined_df['Base_Week']

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2450555012.py in <cell line: 0>()
----> 1 combined_df['Weeks_Passed'] = combined_df['Weeks'] - combined_df['Base_Week']

NameError: name 'combined_df' is not defined

## === cell 13
combined_df.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3289935698.py in <cell line: 0>()
----> 1 combined_df.head()

NameError: name 'combined_df' is not defined

## === cell 14
MIN_MAX_SCALER.fit(combined_df[combined_df['FROM'] == 'train'][['Weeks_Passed', 'FVC', 'Percent', 'Age']])

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014122050.py in <cell line: 0>()
----> 1 MIN_MAX_SCALER.fit(combined_df[combined_df['FROM'] == 'train'][['Weeks_Passed', 'FVC', 'Percent', 'Age']])

NameError: name 'combined_df' is not defined

## === cell 15
combined_df.tail()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1983262329.py in <cell line: 0>()
----> 1 combined_df.tail()

NameError: name 'combined_df' is not defined

## === cell 16
combined_df[['Weeks_Passed', 'Base_FVC', 'Base_Percent', 'Base_Age']] = MIN_MAX_SCALER.transform(combined_df[['Weeks_Passed', 'Base_FVC', 'Base_Percent', 'Base_Age']])

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1565968371.py in <cell line: 0>()
----> 1 combined_df[['Weeks_Passed', 'Base_FVC', 'Base_Percent', 'Base_Age']] = MIN_MAX_SCALER.transform(combined_df[['Weeks_Passed', 'Base_FVC', 'Base_Percent', 'Base_Age']])

NameError: name 'combined_df' is not defined

## === cell 17
combined_df['Sex'] = pd.Categorical(combined_df['Sex'], categories=SEX_COLUMNS)
combined_df['SmokingStatus'] = pd.Categorical(combined_df['SmokingStatus'], categories=SMOKING_STATUS_COLUMNS)
combined_df = combined_df.join(pd.get_dummies(combined_df['Sex']))
combined_df = combined_df.join(pd.get_dummies(combined_df['SmokingStatus']))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/175916908.py in <cell line: 0>()
      1 # convert categoricals into dummies
----> 2 combined_df['Sex'] = pd.Categorical(combined_df['Sex'], categories=SEX_COLUMNS)
      3 combined_df['SmokingStatus'] = pd.Categorical(combined_df['SmokingStatus'], categories=SMOKING_STATUS_COLUMNS)
      4 combined_df = combined_df.join(pd.get_dummies(combined_df['Sex']))
      5 combined_df = combined_df.join(pd.get_dummies(combined_df['SmokingStatus']))

NameError: name 'combined_df' is not defined

## === cell 18
combined_df.drop_duplicates(inplace=True)
combined_df.reset_index(drop=True)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/275668401.py in <cell line: 0>()
----> 1 combined_df.drop_duplicates(inplace=True)
      2 combined_df.reset_index(drop=True)

NameError: name 'combined_df' is not defined

## === cell 19
class PulmonaryDataset(Dataset):
    def __init__(self, df, FV, test=False):
        self.df = df
        self.test = test
        self.FV = FV

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.df[self.FV].iloc[idx].values),
            'target': torch.tensor(self.df['FVC'].iloc[idx])
        }

    def __len__(self):
        return len(self.df)

## === cell 20
class PulmonaryModel(nn.Module):
    def __init__(self, in_features=9, out_quantiles=3):
        super(PulmonaryModel, self).__init__()
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_quantiles)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## === cell 21
new_test_df = combined_df[combined_df['FROM'] == 'test'].reset_index(drop=True)
test_dataset = PulmonaryDataset(new_test_df, FV)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=10,
    drop_last=False,
    num_workers=2
)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/46849969.py in <cell line: 0>()
----> 1 new_test_df = combined_df[combined_df['FROM'] == 'test'].reset_index(drop=True)
      2 test_dataset = PulmonaryDataset(new_test_df, FV)
      3 
      4 test_data_loader = DataLoader(
      5     test_dataset,

NameError: name 'combined_df' is not defined

## === cell 22
models = []
for fold in range(5):
    model = PulmonaryModel(len(FV))
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"model_fold_{fold}.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    models.append(model)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1411795972.py in <cell line: 0>()
      2 for fold in range(5):
      3     model = PulmonaryModel(len(FV))
----> 4     checkpoint = torch.load(os.path.join(MODEL_DIR, f"model_fold_{fold}.pt"))
      5     model.load_state_dict(checkpoint['model_state_dict'])
      6     model.to(DEVICE)

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/osicqrmodel/model_fold_0.pt'

## === cell 23
avg_preds = np.zeros((len(test_dataset), len(QUANTILES)))
with torch.no_grad():
    for model in models:
        preds = []
        for j, test_data in enumerate(test_data_loader):
            features = test_data['features']
            targets = test_data['target']

            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).float()

            out = model(features)
            preds.append(out)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        avg_preds += preds
    avg_preds /= len(models)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1813911884.py in <cell line: 0>()
----> 1 avg_preds = np.zeros((len(test_dataset), len(QUANTILES)))
      2 with torch.no_grad():
      3     for model in models:
      4         preds = []
      5         for j, test_data in enumerate(test_data_loader):

NameError: name 'test_dataset' is not defined

## === cell 24
avg_preds

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2513964544.py in <cell line: 0>()
----> 1 avg_preds

NameError: name 'avg_preds' is not defined

## === cell 25
new_test_df['FVC'] = avg_preds[:, 1]
new_test_df['Confidence'] = np.abs(avg_preds[:, 2] - avg_preds[:, 0])

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476738988.py in <cell line: 0>()
----> 1 new_test_df['FVC'] = avg_preds[:, 1]
      2 new_test_df['Confidence'] = np.abs(avg_preds[:, 2] - avg_preds[:, 0])

NameError: name 'avg_preds' is not defined

## === cell 26
new_test_df.head(25)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1183828827.py in <cell line: 0>()
----> 1 new_test_df.head(25)

NameError: name 'new_test_df' is not defined

## === cell 27
new_test_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/594418837.py in <cell line: 0>()
----> 1 new_test_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index=False)

NameError: name 'new_test_df' is not defined
