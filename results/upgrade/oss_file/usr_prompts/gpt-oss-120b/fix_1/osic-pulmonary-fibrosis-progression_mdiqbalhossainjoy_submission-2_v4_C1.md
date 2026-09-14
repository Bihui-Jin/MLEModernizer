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

No external packages required in the script and installed.

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

-7.677333017750183

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
import pydicom # this one is to read the dicom files 
import scipy.ndimage
import matplotlib.pyplot as plt 
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm 


import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology 



## === cell 1
def csv_split (data, v, t):
    
    data.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
    
    drop_patientID = ['']
    for i in drop_patientID:
        ind=data.Patient[data.Patient == i ].index.tolist()
        for j in ind:
            data=data.drop([j], axis=0)
    data.reset_index(inplace = True, drop = True)
    
    unique_patient=data.Patient.unique()
    unique_patient_val=unique_patient[-v:]
    unique_patient_test=unique_patient[-(v+t):-v]
    unique_patient_train=unique_patient[:-(v+t)]
    
    valid=pd.DataFrame()
    for id in unique_patient_val:
        valid_x=data.loc[data['Patient']==id]
        valid=pd.concat([valid,valid_x])
    test=pd.DataFrame()
    for id in unique_patient_test:
        test_x=data.loc[data['Patient']==id]
        test=pd.concat([test,test_x])
    train=pd.DataFrame()
    for id in unique_patient_train:
        train_x=data.loc[data['Patient']==id]
        train=pd.concat([train,train_x])
    
    valid.reset_index(inplace = True, drop = True)
    test.reset_index(inplace = True, drop = True)
    train.reset_index(inplace = True, drop = True)
    
    return train, valid, test

def csv_preprocess (data):
    
    data['Healthy-FVC']=round((data['FVC']*100)/data['Percent'])
    FE=[]
    FE.append('Healthy-FVC')
    
    COLS = ['Sex','SmokingStatus']
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)
    
    data =  data[['Patient','Weeks','FVC','Age']+FE]
    
    FE1=['Male','Female','Ex-smoker','Never smoked','Currently smokes']
    rename_col={'Weeks':'base_Weeks','FVC':'base_FVC'}
    data=data.rename(columns=rename_col)
    
    data.base_Weeks+=12
    npData=pd.DataFrame(columns=['Patient','base_Weeks','base_FVC','Age','Healthy-FVC']+FE1+['Week','actual_FVC'])
    
    for pid in data['Patient'].unique():
        weeks=data.loc[data['Patient']==pid].base_Weeks
        fvc = data.loc[data['Patient']==pid].base_FVC
        index = data.loc[data['Patient']==pid].index
        weeks.reset_index(inplace = True, drop = True)
        fvc.reset_index(inplace = True, drop = True)
        for j in index:
            for k in range(len(weeks)):
                if (weeks[k] == data.at[j,'base_Weeks']):
                    continue
                else:
                    npData=pd.concat([npData,data.loc[data.index==j]], sort=False)
                    npData.iloc[-1, npData.columns.get_loc('Week')]=weeks[k]
                    npData.iloc[-1, npData.columns.get_loc('actual_FVC')]=fvc[k]
    npData.reset_index(inplace = True, drop = True)
    npData=npData.fillna(0)
    
    npData=sklearn.utils.shuffle(npData)
    npData.reset_index(inplace = True, drop = True)
    
    return npData

## === cell 2
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values = False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)

## === cell 3
def sigma_generator (data):
    confidence=np.arange(70,1000,1)
    data['actual_sigma']=np.nan
    FVC=data['actual_FVC'].values
    Pred=data['Prediction'].values
    for j in range(len(FVC)):
        score=laplace_log_likelihood(FVC[j], Pred[j], confidence, return_values = True)
        ind=np.where(score == score.max())
        i = int(ind[0])
        actual_sigma=confidence[i]
        data.at[j, 'actual_sigma']= actual_sigma
    return data

## === cell 4
class DATA(nn.Module):
	def __init__(self):
		super(DATA, self).__init__()
		
		self.layer1 = nn.Linear(10,64)
		self.layer2 = nn.ReLU()
		self.layer3 = nn.Linear(64,128)
		self.layer4 = nn.ReLU()
		self.layer5 = nn.Linear(128,256)
		self.layer6 = nn.ReLU()
		self.layer7 = nn.Linear(256,512)
		self.layer8 = nn.ReLU()
		self.layer9 = nn.Linear(512,512)
		self.layer10 = nn.ReLU()
		self.layer11 = nn.Linear(512,512)
		self.layer12 = nn.ReLU()
		self.layer13 = nn.Linear(512, 512)
		self.layer14 = nn.ReLU()
		self.layer15 = nn.Linear(512,128)
		self.layer16 = nn.ReLU()
		self.layer17 = nn.Linear(128,64)
		self.layer18 = nn.ReLU()
		self.layer19 = nn.Linear(64,1)
		self.layer20 = nn.ELU()

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x =x1= self.layer7(x)
		x = self.layer8(x)
		x = self.layer9(x)
		x = self.layer10(x)
		x = self.layer11(x)

		x = self.layer12(x)
		x = self.layer13(x)
		x = self.layer14(x)
		x = x+x1  
		x = self.layer15(x)
		x = self.layer16(x)
		x = self.layer17(x)
		x = self.layer18(x)
		x = self.layer19(x)
		x = self.layer20(x)
		return x

class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1=nn.Sequential(
                        nn.Linear(10,64),
                        nn.ReLU(),
                        nn.Linear(64,118),
                        nn.ReLU()
                        )
        self.data_net2=nn.Sequential(
                        nn.Linear(128,256),
                        nn.ReLU(),
                        nn.Linear(256,502),
                        nn.ReLU()
                        )
        self.data_net3=nn.Sequential(
                        nn.Linear(512,256),
                        nn.ReLU(),
                        nn.Linear(256,118),
                        nn.ReLU()
                        )
        self.data_net4=nn.Sequential(
                        nn.Linear(748,64),
                        nn.ReLU(),
                        nn.Linear(64,1),
                        nn.ReLU()
                        )

    def forward(self, data_i):
        out1 = self.data_net1(data_i)
        out2 = torch.cat((data_i,out1), dim=-1)
        out2 = self.data_net2(out2)
        out3 = torch.cat((data_i,out2), dim=-1)
        out3 = self.data_net3(out3)
        out4 = torch.cat((data_i,out1,out2,out3), dim=-1)
        out = self.data_net4(out4)
        return out

## === cell 5

def train_data_net(epochs, batch_size, npTrain,npValid, model, train_device ='cpu'):


    x_train_values_df = npTrain[['base_Weeks', 'base_FVC', 'Age', 'Male', 'Female', 'Ex-smoker','Never smoked', 'Currently smokes', 'Week', 'Healthy-FVC']]
    x_train_values = x_train_values_df.values # ndarray of train metadata 
    y_train_values = npTrain['actual_FVC'].values # ndarray of metadata label 

    x_valid_values_df = npValid[['base_Weeks', 'base_FVC', 'Age', 'Male', 'Female', 'Ex-smoker','Never smoked', 'Currently smokes', 'Week', 'Healthy-FVC']] # dataframw without patientId 
    x_valid_values = x_valid_values_df.values # ndarray of train metadata 
    y_valid_values = npValid['actual_FVC'].values # ndarray of metadata label
    
    if train_device =='cuda':
        device = torch.device("cuda")
        model.to(device)

    
    for epoch in range(epochs):
        torch.backends.cudnn.benchmark = True

        n = len(x_train_values)
        model.train()
        Steps = (n-1)// batch_size +1
        pbar = tqdm(range(Steps), total= Steps)
        for i in pbar:     
            start_i = i * batch_size
            end_i = start_i + batch_size  
            xb_meta =  torch.tensor(x_train_values[start_i:end_i]).float()
            Y_target = torch.tensor(y_train_values[start_i:end_i]).float().unsqueeze(1)

            if train_device == 'cuda':
                xb_meta = xb_meta.cuda()
                Y_target = Y_target.cuda()
            prediction = model(xb_meta)
            loss = compute_loss(prediction, Y_target)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy =(1- ((prediction- Y_target)/Y_target).abs())

            s = ('Epochs: %5d/%d , Steps: %8d/%d , train_loss: %5.3f  ,trian_accuracy: %5.3f'%\
                  (epoch, epochs, i, Steps, loss.data, accuracy.data.item()))
            pbar.set_description(s)
            optimizer.zero_grad()
            del prediction
             
            
             


        val_acc_total = 0.
        n = len(x_valid_values)
        Steps = (n-1)// batch_size +1
        pbar = tqdm(range(Steps), total= Steps)
        model.eval()
        for i in pbar:  #range((n-1)//batch_size +1)
            start_i = i * batch_size
            end_i = start_i + batch_size 
            xb_meta =  torch.tensor(x_valid_values[start_i:end_i]).float()
            Y_target = torch.tensor(y_valid_values[start_i:end_i]).float().unsqueeze(1)
            
            if train_device == 'cuda':
                xb_meta = xb_meta.cuda()
                Y_target = Y_target.cuda()
            prediction = model(xb_meta)
            loss = compute_loss(prediction, Y_target)
            with torch.no_grad():
                accuracy =(1- ((prediction- Y_target)/Y_target).abs())
            

            s = ('Epochs: %5d/%d , Steps: %8d/%d , val_loss: %5.3f  ,val_accuracy: %5.3f'%\
                  (epoch, epochs, i, Steps, loss.data, accuracy.data.item()))
            pbar.set_description(s)
            
            val_acc_total += accuracy.data.item()
            del prediction
        avg_val_acc = (val_acc_total)/n 
        print('Average Validation accuracy:', avg_val_acc)
             


def train_sigma_net(epochs, batch_size, npTrain, npValid, model, train_device ='cpu'):


    
    x_train_values_df = npTrain[['base_Weeks','base_FVC','Age','Male','Female','Ex-smoker','Never smoked','Currently smokes','Healthy-FVC','Prediction']]
    x_train_values = x_train_values_df.values
    y_train_values = npTrain['actual_sigma'].values

    x_valid_values_df = npValid[['base_Weeks','base_FVC','Age','Male','Female','Ex-smoker','Never smoked','Currently smokes','Healthy-FVC','Prediction']]
    x_valid_values = x_valid_values_df.values
    y_valid_values = npValid['actual_sigma'].values
    
    if train_device =='cuda':
        device = torch.device("cuda")
        model.to(device)

    
    for epoch in range(epochs):
        torch.backends.cudnn.benchmark = True

        n = len(x_train_values)
        model.train()
        Steps = (n-1)// batch_size +1
        pbar = tqdm(range(Steps), total= Steps)
        for i in pbar:     #range((n-1)// batch_size +1):
            start_i = i * batch_size
            end_i = start_i + batch_size  
            xb_meta =  torch.tensor(x_train_values[start_i:end_i]).float()
            Y_target = torch.tensor(y_train_values[start_i:end_i]).float().unsqueeze(1)

            if train_device == 'cuda':
                xb_meta = xb_meta.cuda()
                Y_target = Y_target.cuda()
            prediction = model(xb_meta)
            loss = ((prediction- Y_target)/Y_target).abs()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy =(1- ((prediction- Y_target)/Y_target).abs())

            s = ('Epochs: %5d/%d , train_loss: %5.3f  ,trian_accuracy: %5.3f'%\
                  (epoch, epochs,  loss.data, accuracy.data.item()))
            pbar.set_description(s)
            optimizer.zero_grad()
            del prediction
             
            
             


        val_loss=0
        val_acc_total = 0.
        n = len(x_valid_values)
        Steps = (n-1)// batch_size +1
        pbar = tqdm(range(Steps), total= Steps)
        model.eval()
        for i in pbar:  #range((n-1)//batch_size +1)
            start_i = i * batch_size
            end_i = start_i + batch_size 
            xb_meta =  torch.tensor(x_valid_values[start_i:end_i]).float()
            Y_target = torch.tensor(y_valid_values[start_i:end_i]).float().unsqueeze(1)
            
            if train_device == 'cuda':
                xb_meta = xb_meta.cuda()
                Y_target = Y_target.cuda()
            prediction = model(xb_meta)
            loss = ((prediction- Y_target)/Y_target).abs()
            with torch.no_grad():
                accuracy =(1- ((prediction- Y_target)/Y_target).abs())
            

            s = ('Epochs: %5d/%d , val_loss: %5.3f  ,val_accuracy: %5.3f'%\
                  (epoch, epochs,  loss.data, accuracy.data.item()))
            pbar.set_description(s)
            
            val_loss += loss.data.item()
            val_acc_total += accuracy.data.item()
            del prediction
        avg_loss = val_loss/n
        avg_val_acc = (val_acc_total)/n 
        print('Average Validation accuracy:', avg_val_acc)
        print('Average Validation loss:', avg_loss)
             


## === cell 6
def make_eval_data(npEval, model):
    x_features = npEval[['base_Weeks', 'base_FVC', 'Age', 'Male', 'Female', 'Ex-smoker','Never smoked', 'Currently smokes', 'Week', 'Healthy-FVC']]
    x_features = torch.tensor(x_features.values).float()
    patientsID = npEval['Patient'].values 
    
    predictions = []
    for x_feature in x_features:
        x_feature = x_feature.unsqueeze(0)
        prediction = model(x_feature)
        predictions.append(prediction.data.item())
    npEval['Prediction'] = predictions
    npEval.reset_index(inplace = True)
    return npEval



def make_eval_sigma(npEval, model):
    x_features = npEval[['base_Weeks','base_FVC','Age','Male','Female','Ex-smoker','Never smoked','Currently smokes','Healthy-FVC','Prediction']]

    x_features = torch.tensor(x_features.values).float()
    patientsID = npEval['Patient'].values 
    
    confidences = []

    for i in range(x_features.shape[0]):
        x_feature  = x_features[i]
        x_feature = x_feature.unsqueeze(0)
        confidence = model(x_feature)
        confidences.append(confidence.data.item())
        
    npEval['confidence'] = confidences
    return npEval

## === cell 7
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission['Patient']=submission['Patient_Week'].apply(lambda x:x.split('_')[0])
submission['Weeks']=submission['Patient_Week'].apply(lambda x:x.split('_')[1]).astype(int)

submission.Weeks += 12


testdf = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
merge=pd.merge(testdf,submission,on=['Patient'],how='left').sort_values(['Weeks_y','Patient']).reset_index(drop=True)
merge=merge.drop(['FVC_y'],axis=1)
merge=merge.rename(columns={'FVC_x':'base_FVC','Weeks_y':'Week','Weeks_x':'base_Weeks'})

del testdf
del submission

testdf=merge.loc[:,['Patient','base_Weeks','base_FVC','Percent','Age','Sex','SmokingStatus','Week']]
submission=merge.loc[:,['Patient_Week','base_FVC','Confidence']]
submission=submission.rename(columns={'base_FVC':'FVC'})

## === cell 8
data = testdf.copy()
data['Healthy-FVC']=round((data['base_FVC']*100)/data['Percent'])
FE=[]
FE.append('Healthy-FVC')

COLS = ['Sex','SmokingStatus']
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)
FE1=['Male','Female','Ex-smoker','Never smoked','Currently smokes']
npData=pd.DataFrame(columns=['Patient','base_Weeks','base_FVC','Age','Healthy-FVC']+FE1+['Week'])
npData=npData.append(data)
npData=npData.fillna(0)

del testdf
testdf = npData[['Patient','base_Weeks','base_FVC','Age','Healthy-FVC','Male','Female','Ex-smoker','Never smoked','Currently smokes','Week']]
   

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1290125304.py in <cell line: 0>()
     12 FE1=['Male','Female','Ex-smoker','Never smoked','Currently smokes']
     13 npData=pd.DataFrame(columns=['Patient','base_Weeks','base_FVC','Age','Healthy-FVC']+FE1+['Week'])
---> 14 npData=npData.append(data)
     15 npData=npData.fillna(0)
     16 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 10
model_FVC = DATA()
model_FVC.load_state_dict(torch.load('../input/metadatapreweights/metadata_checkpoint.pth'))
model_FVC.eval()

test_inp_sigma = make_eval_data(testdf.copy(), model_FVC)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2955171450.py in <cell line: 0>()
      1 model_FVC = DATA()
----> 2 model_FVC.load_state_dict(torch.load('../input/metadatapreweights/metadata_checkpoint.pth'))
      3 model_FVC.eval()
      4 # for the validation inputset of sigma
      5 # ndf = pd.concat([npValid,npTest])

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/metadatapreweights/metadata_checkpoint.pth'

## === cell 11
model_sigma = SIGMA()
model_sigma.load_state_dict(torch.load('../input/sigmapreweight/sigma_preweight.pth'))
model_sigma.eval()


test_confidence_df = make_eval_sigma(test_inp_sigma.copy(), model_sigma)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2404466554.py in <cell line: 0>()
      1 model_sigma = SIGMA()
----> 2 model_sigma.load_state_dict(torch.load('../input/sigmapreweight/sigma_preweight.pth'))
      3 model_sigma.eval()
      4 
      5 

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/sigmapreweight/sigma_preweight.pth'

## === cell 12
submission.loc[:, 'FVC']=test_inp_sigma.Prediction
submission.loc[:, 'Confidence']= test_confidence_df.confidence



submission.to_csv('submission.csv',index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1539953585.py in <cell line: 0>()
----> 1 submission.loc[:, 'FVC']=test_inp_sigma.Prediction
      2 submission.loc[:, 'Confidence']= test_confidence_df.confidence
      3 
      4 
      5 #print(submission.tail())

NameError: name 'test_inp_sigma' is not defined
