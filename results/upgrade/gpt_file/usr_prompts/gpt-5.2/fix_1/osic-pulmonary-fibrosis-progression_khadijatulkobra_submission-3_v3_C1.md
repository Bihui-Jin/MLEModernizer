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

3.9

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

-6.950531477454692

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
import pydicom # this one is to read the dicom files 
import os 
import scipy.ndimage
import matplotlib.pyplot as plt 
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm 


import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology 
from sklearn.preprocessing import normalize

from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset 

## === cell 1
TRAIN_FOLDER = '../data/train'





def load_scan(path): # Here path == (../input/osic-pulmonary-fibrosis-progression/train/patientId)
	try:
		slices = [pydicom.dcmread(path+os.sep+s) for s in os.listdir(path)]
		slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
	except:
		files = os.listdir(path)
		files.sort()
		slices = [pydicom.dcmread(path+os.sep+s) for s in files]


	return slices 


'''
patientIDs = ['ID00007637202177411956430', 'ID00009637202177434476278']
'''
def make_dict_with_slices(patientIDs): # this will be very huge memory consuming . don't use it 
	patient_slices_dict = {}
	failed_slices_patiendIDs = []

	for patientID in patientIDs:
		path = TRAIN_FOLDER + os.sep + patientID

		try:
			patient_slices_dict[patientID] = load_scan(path)
		except:
			failed_slices_patiendIDs.append(patientID)

	return patient_slices_dict, failed_slices_patiendIDs 





'''
here slices is a pydicom.dcmread() file combinations 
'''
def get_pixels_hu(slices):
	image = np.stack([s.pixel_array for s in slices])
	image = image.astype(np.int16)
	try:
		image[image <= -2000] = 0
		for slice_number in range(len(slices)):
			intercept = slices[slice_number].RescaleIntercept
			slope = slices[slice_number].RescaleSlope

			if slope !=1:
				image[slice_number] =slope *image[slice_number].astype(np.float64)
				image[slice_number] = image[slice_number].astype(np.int16)
			image[slice_number] += np.int16(intercept)
	except:
		print('HU conversion Failed!!')

	return np.array(image, dtype = np.int16)


'''
Here input is either pydicom.dcmread class or numpy3d array 
'''
def plot_show_slice(slices):
	if not isinstance(slices, type(np.array([]))):
		first_patient_pixels = get_pixels_hu(slices) ## conversion to np array and HU unit 
	else:
		first_patient_pixels = slices 

	print('Number of Total Slices in this Scan:', len(slices))
	try:
		print('Shape of the the Image is:', slices.shape[1], slices.shape[2])
	except:
		print('Shape of the the Image is:BLANK')
	fig = plt.figure(figsize=(10,10))  
	for i,slice in enumerate(first_patient_pixels[:16]):
		y=fig.add_subplot(4,4, i+1)
		y.imshow(slice, cmap='gray')
	plt.show()

'''
Here input slices are numpy array not a pydicom.dcmread class 
returned a numpy 3darray 
'''
def resize_along_zaxis(slices, target_dimension=30):
	present_dimension = len(slices)
	if target_dimension == present_dimension:
		return slices

	zoom_factor = float(target_dimension)/float(present_dimension)
	resize_image=scipy.ndimage.zoom(slices, [zoom_factor, 1., 1.])
	
	return resize_image

'''
input: 3d numpy array
output: 3d numpy array 

'''

def resize_along_allaxis(slices, target_dimensionZ=30, target_dimensionY= 100, target_dimensionX= 100):
	present_dimensionZ, present_dimensionY, present_dimensionX = slices.shape[0], slices.shape[1] ,slices.shape[2]
	if target_dimensionZ == present_dimensionZ and\
            target_dimensionY == present_dimensionY and target_dimensionX == present_dimensionX:        
		return slices
	zoom_factorZ = float(target_dimensionZ)/float(present_dimensionZ)
	zoom_factorY = float(target_dimensionY)/float(present_dimensionY)
	zoom_factorX = float(target_dimensionX)/float(present_dimensionX)
                         
	resize_image=scipy.ndimage.zoom(slices, [zoom_factorZ,zoom_factorY , zoom_factorX], mode='nearest')
	
	return resize_image


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def image_normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
'''
patient_folder = './osic-pulmonary-fibrosis-progression/train'
			### where the patientId folder and their slices in it 
output_folder = 'Relative path with respect to PWD where data will be saved'
'''
def save_array(patientID_folder, output_folder,Z=100,Y=200,X=200): #patientID_folder, output_folder
	patientIDs = os.listdir(patientID_folder)
	patientIDs.sort()
	Save_dir = output_folder # './trainset'
	if not os.path.exists(Save_dir):
		print('The output directory doesnt exists')
		raise Exception 
		
	for i,patientID in enumerate(patientIDs):
		path = TRAIN_FOLDER + os.sep + patientIDs[i]
		slices = load_scan(path)
		try:
			image_array = get_pixels_hu(slices) # HU unit conversion + nparray
			ctimage_resizedAll = resize_along_allaxis(image_array, target_dimensionX=X,
													target_dimensionY=Y,target_dimensionZ=Z)
			image=(image_normalize(ctimage_resizedAll)*255.0).astype('uint8')
            
			np.save(Save_dir+os.sep+patientID+'.npy', image)
		except:
			print('PatientId:%s couldnt be save and converted'%(patientID))

def load_array(path): ### path='./traindataset/ID00052637202186188008618.npy'
	try:
		if(path.endswith('.npy')):
			image_array = np.load(path)
		else:
			path= path+'.npy'
			image_array = np.load(path)
	except:
		
		print('The file in the Path:%s doesnetexists!!'%(path.split('\\')[-1].split('/')[-1].split('.')[0]))
		return []
	return image_array

def read_image(dir_name,patientid,Z=100,Y=200,X=200): #patientID_folder, output_folder
	path = dir_name + os.sep + patientid
	slices = load_scan(path)
	try:
		image_array = get_pixels_hu(slices) # HU unit conversion + nparray
		ctimage_resizedAll = resize_along_allaxis(image_array, target_dimensionX=X,
												target_dimensionY=Y,target_dimensionZ=Z)
		image=(image_normalize(ctimage_resizedAll)*255.0).astype('uint8')
		return image
	except:
		print('PatientId:%s couldnt be converted'%(patientid))

## === cell 2
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
    data = data.sort_values(['Patient','Weeks'], ascending=True).reset_index(drop=True)
    
    FE1=['Male','Female','Ex-smoker','Never smoked','Currently smokes']
    rename_col={'Weeks':'base_Weeks','FVC':'base_FVC'}
    data=data.rename(columns=rename_col)
    
    npData=pd.DataFrame(columns=['Patient','base_Weeks','base_FVC','Age']+FE1+['Week','Healthy-FVC','actual_FVC'])
    
    for pid in data['Patient'].unique():
        weeks=data.loc[data['Patient']==pid].base_Weeks
        fvc = data.loc[data['Patient']==pid].base_FVC
        index = data.loc[data['Patient']==pid].index
        weeks.reset_index(inplace = True, drop = True)
        fvc.reset_index(inplace = True, drop = True)
        for k in range(len(weeks)):
            npData=pd.concat([npData,data.loc[data.index==index[0]]], sort=False)
            npData.iloc[-1, npData.columns.get_loc('Week')]=weeks[k]
            npData.iloc[-1, npData.columns.get_loc('actual_FVC')]=fvc[k]
    npData.reset_index(inplace = True, drop = True)
    npData=npData.fillna(0)
    
    npData['Week']=npData['Week']-npData['base_Weeks']
    npData['base_Weeks']=0.0
    
    return npData

## === cell 3
class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class ds_3d_conv(nn.Module):
	def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer):
		super(ds_3d_conv, self).__init__()
		self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
		self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

	def forward(self, x):
		out = self.depthwise(x)
		out = self.pointwise(out)
		return out

class SIGMA(nn.Module):
	def __init__(self):
		super(SIGMA, self).__init__()
		self.data_net1=nn.Sequential(
						nn.Linear(41,64),
						nn.ReLU(),
						nn.Linear(64,119),
						nn.ReLU()
						)
		self.data_net2=nn.Sequential(
						nn.Linear(128,256),
						nn.ReLU(),
						nn.Linear(256,503),
						nn.ReLU()
						)
		self.data_net3=nn.Sequential(
						nn.Linear(512,256),
						nn.ReLU(),
						nn.Linear(256,119),
						nn.ReLU()
						)
		self.data_net4=nn.Sequential(
						nn.Linear(750,256),
						nn.ReLU(),
						nn.Linear(256,64),
						nn.ReLU(),
						nn.Linear(64,3),
						nn.ReLU()
						)

	def forward(self, data_i, image_o):
		x = torch.cat((data_i, image_o), dim=-1)
		out1 = self.data_net1(x)
		out2 = torch.cat((data_i,out1), dim=-1)
		out2 = self.data_net2(out2)
		out3 = torch.cat((data_i,out2), dim=-1)
		out3 = self.data_net3(out3)
		out4 = torch.cat((data_i,out1,out2,out3), dim=-1)
		out = self.data_net4(out4)
		return out

        



class IMAGE(nn.Module):
	def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=16, dropout=True):
		super(IMAGE, self).__init__()
		n_layer = len(channel_number)
		self.feature_extractor = nn.Sequential()
		for i in range(n_layer):
			if i == 0:
				in_channel = 1
			else:
				in_channel = channel_number[i-1]
			out_channel = channel_number[i]
			if i < n_layer-1:
				self.feature_extractor.add_module('conv_%d' % i,
												  self.conv_layer(in_channel,
																  out_channel,
																  maxpool=True,
																  kernel_size=3,
																  padding=1,
																  kernels_per_layer=1))
			else:
				self.feature_extractor.add_module('conv_%d' % i,
												  self.conv_layer(in_channel,
																  out_channel,
																  maxpool=False,
																  kernel_size=1,
																  padding=0,
																  kernels_per_layer=1))
		self.classifier = nn.Sequential()
		if dropout is True:
			self.classifier.add_module('dropout', nn.Dropout(0.5))
		i = n_layer
		in_channel = channel_number[-1]
		out_channel = output_dim
		self.classifier.add_module('conv_%d' % i,
								   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1)
								  )
		self.flat=nn.Sequential(
			Flatten(),
			nn.Linear(1728,512),
			nn.ReLU(),
			nn.Linear(512,128),
			nn.ReLU(),
			nn.Linear(128,32),
			nn.ReLU()
		)
	@staticmethod
	def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1, kernels_per_layer=1, maxpool_stride=2):
		if maxpool is True:
			layer = nn.Sequential(
				ds_3d_conv(in_channel, out_channel, kernel_size, padding, kernels_per_layer),
				nn.BatchNorm3d(out_channel),
				nn.MaxPool3d(2, stride=maxpool_stride),
				nn.ReLU(),
			)
		else:
			layer = nn.Sequential(
				ds_3d_conv(in_channel, out_channel, kernel_size, padding, kernels_per_layer),
				nn.BatchNorm3d(out_channel),
				nn.ReLU()
			)
		return layer

	def forward(self, image_i):
		image_o = self.feature_extractor(image_i)
		image_o = self.classifier(image_o)
		image_o = self.flat(image_o)
		return image_o
	
class Combined_NET(nn.Module):
	def __init__(self):
		super(Combined_NET, self).__init__()
		self.image = IMAGE()
		self.data = SIGMA()
		
	def forward(self, image_i, data_i):
		image_o = self.image(image_i)
		data_o = self.data(data_i,image_o)
		return data_o

## === cell 4
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype= torch.float32)

def score(y_true, y_pred):
	sigma = y_pred[:, 2] - y_pred[:, 0]
	fvc_pred = y_pred[:, 1]
	
	c1_same_shape = torch.ones(sigma.size(),device=y_pred.device)*C1 
	sigma_clip = torch.max(sigma, c1_same_shape)
	delta = (y_true[:, 0] - fvc_pred).abs()

	c2_same_shape = torch.ones(delta.size(),device=y_pred.device)*C2 
	delta = torch.min(delta, c2_same_shape)

	sq2 = torch.tensor(2.).sqrt()
	metric = (delta / sigma_clip)*sq2 + (sigma_clip* sq2).log()
	return (metric).mean()

def qloss(y_true, y_pred):
	qs = [0.25, 0.50, 0.75]
	q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
	e = y_true - y_pred # [9] - [5, 7, 8] = [4,2, 1]
	v = torch.max(q*e, (q-1)*e) # [14,30,66], [14,30,66]
	return v.mean()



def quartile_loss(y_true, y_pred, _lambda = 0.65): # 0.65 
	loss = _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred) # 0.35
	return loss

## === cell 6
def make_eval_data(npEval, model, device = 'cuda'):
    x_features = npEval[['base_FVC','Age','Male','Female','Ex-smoker','Never smoked','Currently smokes','Week','Healthy-FVC']]
    x_features = torch.tensor(x_features.values).float()
    x_patientids_name = npEval[['Patient']].values
    
    unique_patients = npEval.Patient.unique()
    loaded_images = {}
    dir_name_of_patientid= '../input/osic-pulmonary-fibrosis-progression/test'
    
    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(dir_name_of_patientid, unique_patient, Z=100,Y=200,X=200)
        
    
    
    if torch.cuda.is_available() and device == 'cuda':
        model.to('cuda')
    model.eval()
    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        x_image = loaded_images[patientid[0]]
        
        x_image = torch.tensor(x_image, dtype = torch.float32).unsqueeze(0).unsqueeze(0) # one for channel another for batch 
        x_feature = x_features[i] # taking the feature for the particular patientid and week of npEval dataframe 
        x_feature = x_feature.unsqueeze(0)
        
        
        if torch.cuda.is_available() and device == 'cuda':
            x_image = x_image.cuda()
            x_feature = x_feature.cuda()
            
        prediction = model(x_image, x_feature)
        predictions.append(prediction.to('cpu').detach().numpy()[0]) # since it returns batched output , and we use batch_size =1, taking the [0] output 
        
    
    predictions = np.array(predictions) # shape = [none, 3]
    npEval['FVC'] = predictions[:,1]
    npEval['Confidence'] = predictions[:,2] - predictions[:,0]
    
    return npEval

## === cell 7
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission  = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")


## === cell 8
submission['Patient']=submission['Patient_Week'].apply(lambda x:x.split('_')[0])
submission['Weeks']=submission['Patient_Week'].apply(lambda x:x.split('_')[1]).astype(int)
submission=submission.sort_values(by=['Patient','Weeks'], ascending=True ).reset_index(drop=True)


merge=pd.merge(data_test,submission,on=['Patient'],how='left').sort_values(['Patient','Weeks_y']).reset_index(drop=True)
merge=merge.drop(['FVC_y'],axis=1)
merge=merge.rename(columns={'FVC_x':'base_FVC','Weeks_y':'Week','Weeks_x':'base_Weeks'})

del data_test
del submission

data_test=merge.loc[:,['Patient','base_Weeks','base_FVC','Percent','Age','Sex','SmokingStatus','Week']]
submission=merge.loc[:,['Patient_Week','base_FVC','Confidence']]
submission=submission.rename(columns={'base_FVC':'FVC'})

## === cell 9
data = data_test.copy()
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
npData=npData.append(data, sort=True)
npData=npData.fillna(0)

npData['Week']=npData['Week']-npData['base_Weeks']
npData['base_Weeks']=0.0

del data_test, data
data_test = npData[['Patient','base_Weeks','base_FVC','Age','Healthy-FVC','Male','Female','Ex-smoker','Never smoked','Currently smokes','Week']]
del npData   

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1757488488.py in <cell line: 0>()
     12 FE1=['Male','Female','Ex-smoker','Never smoked','Currently smokes']
     13 npData=pd.DataFrame(columns=['Patient','base_Weeks','base_FVC','Age','Healthy-FVC']+FE1+['Week'])
---> 14 npData=npData.append(data, sort=True)
     15 npData=npData.fillna(0)
     16 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 12
model = Combined_NET()
model.load_state_dict(torch.load('../input/test6777/Epoch77_Score6.777315493785974_Acc0.9343944720246575.pth'))

test = make_eval_data(data_test.copy(), model)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/374159366.py in <cell line: 0>()
      1 model = Combined_NET()
----> 2 model.load_state_dict(torch.load('../input/test6777/Epoch77_Score6.777315493785974_Acc0.9343944720246575.pth'))
      3 #model.eval()
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

FileNotFoundError: [Errno 2] No such file or directory: '../input/test6777/Epoch77_Score6.777315493785974_Acc0.9343944720246575.pth'

## === cell 15
submission.loc[:, 'FVC']=test.FVC
submission.loc[:, 'Confidence']= test.Confidence
submission.to_csv('submission.csv',index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/741852275.py in <cell line: 0>()
----> 1 submission.loc[:, 'FVC']=test.FVC
      2 submission.loc[:, 'Confidence']= test.Confidence
      3 submission.to_csv('submission.csv',index=False)

NameError: name 'test' is not defined
