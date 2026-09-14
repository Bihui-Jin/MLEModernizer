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

-8.117

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
import matplotlib.pyplot as plt
import pydicom
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from skimage import morphology
from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.patches as patches


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')


## === cell 2
len(train_csv)


## === cell 3
train_csv.info()


## === cell 4
'''unique_ids=train_csv['Patient'].unique()
week_x=[]
fvc_y=[]
percent=[]
for id in unique_ids:
    week=np.array(train_csv[train_csv['Patient']==id]['Weeks'])
    fvc=np.array(train_csv[train_csv['Patient']==id]['FVC'])
    per=np.array(train_csv[train_csv['Patient']==id]['Percent'])
    week_x.append(week)
    fvc_y.append(fvc)
    percent.append(per)
    
unique_train=pd.DataFrame(train_csv['Patient'].unique(),columns=['Patient'])
unique_train['week_x']=week_x
unique_train['fvc_y']=fvc_y
unique_train['percent']=percent
unique_train.head()'''


## === cell 5
'''X=unique_train['week_x'][0]
Y=unique_train['fvc_y'][0]'''


## === cell 6
'''def using_poly_reg(X,Y,degree=3):
    poly_features=PolynomialFeatures(degree=degree,include_bias=False)
    x_poly=poly_features.fit_transform(X[:,np.newaxis])

    lin_reg=LinearRegression()
    lin_reg.fit(x_poly,Y)

    x_test=np.arange(-12,133)[:,np.newaxis]
    x_test_poly=poly_features.fit_transform(x_test)
    plt.plot(x_test,lin_reg.predict(x_test_poly))
    plt.plot(X,Y)
    #plt.ylim(0,6400)
    #plt.xlim(-12,133)
    plt.grid(True)'''


## === cell 7
'''using_poly_reg(X,Y,degree=3)  #here we can customize the degree of the polynomial so it is better


## === cell 8
'''train_csv['healthy_person_FVC']=(train_csv['FVC']/(train_csv['Percent']/100)).round()'''


## === cell 9
'''train_csv'''


## === cell 10
'''healthy_fvc_info=train_csv.groupby(['Age','Sex','SmokingStatus'])['healthy_person_FVC'].mean().round()'''


## === cell 11
'''plt.plot(healthy_fvc_info[:,'Male','Ex-smoker'],label='male ex smoker')
plt.plot(healthy_fvc_info[:,'Male','Never smoked'],label='male never smoked')
plt.plot(healthy_fvc_info[:,'Male','Currently smokes'],label='male currently smokes')

plt.plot(healthy_fvc_info[:,'Female','Ex-smoker'],label='female ex smoker')
plt.plot(healthy_fvc_info[:,'Female','Never smoked'],label='female never smoked')
plt.plot(healthy_fvc_info[:,'Female','Currently smokes'],label='female currently smokes')

plt.title('healthy fvc related to age,sex and smoking status')
plt.legend()
plt.grid(True)'''


## === cell 12
'''def RForestRegressor(x,y):
    reg=RandomForestRegressor(n_estimators=50)
    reg.fit(x[:,np.newaxis],y)
    x_test=np.arange(0,100)
    y_test=reg.predict(x_test[:,np.newaxis])
    plt.plot(x_test,y_test,label='predicted')
    plt.plot(x,y,label='real')
    plt.grid(True)
    plt.legend()'''


## === cell 13
'''#x=np.array(healthy_fvc_info[:,'Male','Ex-smoker'].index)
#y=np.array(healthy_fvc_info[:,'Male','Ex-smoker'].values)
x=np.array(healthy_fvc_info[:,'Male','Never smoked'].index)
y=np.array(healthy_fvc_info[:,'Male','Never smoked'].values)

RForestRegressor(x,y)'''


## === cell 14
'''X=unique_train['week_x'][0]
Y=unique_train['fvc_y'][0]
RForestRegressor(X,Y)'''


## === cell 15
'''age=train_csv.groupby('Patient')['Age'].unique()'''


## === cell 16
'''for item in age:
    if len(item)==1:
        continue
    else:
        print(item.index)'''


## === cell 17
'''SS=train_csv.groupby('Patient')['SmokingStatus'].unique()
for item in SS:
    if len(item)==1:
        continue
    else:
        print(item)'''


## === cell 18
'''sex=[]
for id in unique_train['Patient']:
    sex.append(train_csv[train_csv['Patient']==id]['Sex'].unique()[0])
    
unique_train['sex']=sex'''


## === cell 19
'''age=[]
for id in unique_train['Patient']:
    age.append(train_csv[train_csv['Patient']==id]['Age'].unique()[0])
    
unique_train['age']=age'''


## === cell 20
'''ss=[]
for id in unique_train['Patient']:
    ss.append(train_csv[train_csv['Patient']==id]['SmokingStatus'].unique()[0])
    
unique_train['smoking-status']=ss'''


## === cell 21
'''unique_train'''


## === cell 22
'''from sklearn.cluster import KMeans

lung=pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/26.dcm')
image=lung.pixel_array
X = image.reshape(-1,1)
#X=image

#good_init=np.array([[-2048],[-1000],[892],[-177],[190]])
#kmeans = KMeans(n_clusters=8,init=good_init,n_init=1).fit(X)

kmeans = KMeans(n_clusters=6).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
plt.imshow(segmented_img)'''


## === cell 23
def fitter(img):
    
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    middle = img[int(col_size/4):int(col_size/4*3),int(row_size/4):int(row_size/4*3)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    return kmeans

lung=pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/26.dcm')
image=lung.pixel_array*lung.RescaleSlope+lung.RescaleIntercept
kmeans=fitter(image)


## === cell 24
def make_lungmask(img,kmeans,display=False):
    image=img
    row_size= img.shape[0]
    col_size = img.shape[1]
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    middle = img[int(col_size/4):int(col_size/4*3),int(row_size/4):int(row_size/4*3)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image


    eroded = morphology.erosion(thresh_img,np.ones([5,5]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
            
    for prop in regions:
        b = prop.bbox
        lung_row=abs((b[2]+b[0])/2-(row_size/2))
        left_lung_col=abs((b[3]+b[1])/2-(col_size/4))
        right_lung_col=abs((b[3]+b[1])/2-(col_size/4)*3)
        
        if lung_row<100 and (left_lung_col<110 or right_lung_col<110):
            good_labels.append(prop.label)
            
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([8,8])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
        
    air=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j]==1:
                air.append(image[i][j])
    if len(air)==0 :
        air_percent=0.0
    else:
        air_percent=abs((sum(air)/len(air))/10).round(4)
    return mask,air_percent


## === cell 26
show_plots=True

if show_plots:
    fig=plt.figure(figsize=(20,20)) 
col=14
row=14 
i=1 
air_percent_dict={}
for id in train_csv['Patient'].unique(): 
    path='../input/osic-pulmonary-fibrosis-progression/train/'+id+'/' 
    filenames=os.listdir(path) 
    fileno=int(len(filenames)/2)
    for item in filenames:
        number=int(item.split('.')[0])
        if number==fileno:
            break
        else:
            continue
    try:
        lung=pydicom.dcmread(path+item) 
        image=lung.pixel_array*lung.RescaleSlope+lung.RescaleIntercept
        mask,air_percent=make_lungmask(image,kmeans,display=False)
        air_percent_dict[id]=air_percent
        if show_plots:
            fig.add_subplot(row,col,i) 
            plt.title(air_percent)
            plt.imshow(mask,cmap='gray')
            plt.grid(False)
            plt.axis(False)
    except: 
        air_percent_dict[id]=np.nan 
    i=i+1 


## === cell 27
for key in air_percent_dict:
    if air_percent_dict[key]<35.0:
        air_percent_dict[key]=np.nan
        print(key)


## === cell 28
train=train_csv[['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']]
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()#sex
train.iloc[:,5]=lb.fit_transform(train.iloc[:,5])
lb2=LabelEncoder()#ss
train.iloc[:,6]=lb2.fit_transform(train.iloc[:,6])


## === cell 29
lung_percent=[]
for id in train['Patient']:
    lung_percent.append(float(air_percent_dict[id]))
train['lung percent']=lung_percent


## === cell 30
train=train.dropna()
train


## === cell 31
train.info()


## === cell 32
len(train)


## === cell 33
test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test.iloc[:,5]=lb.transform(test.iloc[:,5])
test.iloc[:,6]=lb2.transform(test.iloc[:,6])


## === cell 34
air_percent_dict={}
for id in test['Patient'].unique(): 
    path='../input/osic-pulmonary-fibrosis-progression/test/'+id+'/' 
    filenames=os.listdir(path) 
    fileno=int(len(filenames)/2)
    for item in filenames:
        number=int(item.split('.')[0])
        if number==fileno:
            break
        else:
            continue
    try:
        lung=pydicom.dcmread(path+item) 
        image=lung.pixel_array*lung.RescaleSlope+lung.RescaleIntercept
        mask,air_percent=make_lungmask(image,kmeans,display=False)
        air_percent_dict[id]=air_percent
    except: 
        print(id)


## === cell 35
air_percent_dict


## === cell 36
lung_percent_test=[]
for id in test['Patient']:
    lung_percent_test.append(float(air_percent_dict[id]))
test['lung percent']=lung_percent_test
test


## === cell 37
'''def healthy_fvc_predictor(age,sex,smoking_status):
    x=np.array(healthy_fvc_info[:,sex,smoking_status].index)
    y=np.array(healthy_fvc_info[:,sex,smoking_status].values)
    reg=RandomForestRegressor(n_estimators=50)
    reg.fit(x[:,np.newaxis],y)
    return reg.predict([[age]])'''


## === cell 38
def metric(actual_fvc, predicted_fvc, confidence, return_values = False):
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


## === cell 39
def RForestRegressor(x,y):
    reg=RandomForestRegressor(n_estimators=400)
    reg.fit(np.array(x),np.array(y))
    return reg

x=train[['Weeks','Percent','lung percent','SmokingStatus','Sex','Age']]
y=train['FVC']


from sklearn.model_selection import train_test_split
xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)

percent_reg=RForestRegressor(xtrain,ytrain)

preds=percent_reg.predict(np.array(xvalid))
confidence=abs(preds-np.array(yvalid))
print(metric(np.array(yvalid),preds,confidence))


## === cell 40
'''def neural(x):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100,activation='relu',input_shape=x.shape[1:]))
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model

x=train[['Weeks','Percent','lung percent','SmokingStatus','Sex','Age']]
#x=train[['Percent','lung percent','Weeks','Sex']]
y=train['FVC']


from sklearn.model_selection import train_test_split
xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)

model=neural(x)

class metric_callback(tf.keras.callbacks.Callback):
    def __init__(self,metrics,xvalid,yvalid):
        self.metrics=metrics
        self.xvalid=xvalid
        self.yvalid=yvalid
    def on_epoch_end(self,epoch,logs={}):
        preds=self.model.predict(np.array(self.xvalid))
        confidence=abs(preds-np.array(self.yvalid))
        metric=self.metrics(np.array(self.yvalid),preds,confidence)
        print('\r val metrics score :',metric)

history=model.fit(np.array(xtrain),np.array(ytrain),epochs=10,callbacks=[metric_callback(metric,xvalid,yvalid)])'''
        


## === cell 41
def plot_fi(forest,X):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature : %s (%f)" % (f + 1, np.array(X.columns)[indices[f]], importances[indices[f]]))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],color="g", yerr=std[indices])
    plt.xticks(range(X.shape[1]),np.array(X.columns)[indices])
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
plot_fi(percent_reg,x)


## === cell 42
test_csv=test[['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus','lung percent']]
weeks=np.arange(-12,134)
result={}
for id in test_csv['Patient'].unique():
    percent=np.array(test_csv[test_csv['Patient']==id]['Percent'])
    sex=np.array(test_csv[test_csv['Patient']==id]['Sex'])
    age=np.array(test_csv[test_csv['Patient']==id]['Age'])
    ss=np.array(test_csv[test_csv['Patient']==id]['SmokingStatus'])
    lp=np.array(test_csv[test_csv['Patient']==id]['lung percent'])
    percent=np.repeat(percent,len(weeks))
    sex=np.repeat(sex,len(weeks))
    age=np.repeat(age,len(weeks))
    ss=np.repeat(ss,len(weeks))
    lp=np.repeat(lp,len(weeks))
    x=np.concatenate([weeks[:,np.newaxis],percent[:,np.newaxis],lp[:,np.newaxis],ss[:,np.newaxis],sex[:,np.newaxis],age[:,np.newaxis]],axis=1)
    outcome=percent_reg.predict(x)
    result[id]=outcome


## === cell 43
ans_df_list=[]
for id in result:
    ID=np.repeat(id,len(weeks))
    ans=np.concatenate([ID[:,np.newaxis],weeks[:,np.newaxis],result[id][:,np.newaxis]],axis=1)
    ans=pd.DataFrame(ans)
    ans_df_list.append(ans)


## === cell 44
submit=pd.concat(ans_df_list,ignore_index=True)
submit.columns=['Patient','Weeks','FVC']

submit['FVC']=submit['FVC'].astype(float)
submit['Weeks']=submit['Weeks'].astype(int)


## === cell 45
submit


## === cell 46
test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 47
'''healthy_fvc_dict={}
for i in range(len(test_csv)):
    hfvc=healthy_fvc_predictor(test_csv.iloc[i,4],test_csv.iloc[i,5],test_csv.iloc[i,6])
    healthy_fvc_dict[test_csv.iloc[i,0]]=hfvc.ravel()[0]'''


## === cell 48
'''hfvc_list=[]
for i in range(len(submit)):
    hfvc_list.append(healthy_fvc_dict[submit.iloc[i,0]])
    
submit['healthy_fvc']=hfvc_list
submit['Percent']=submit['Percent'].astype(float)
submit['FVC']=(submit['healthy_fvc']*submit['Percent'])/100'''


## === cell 49
confidence_dict={}
for id in submit['Patient'].unique():
    real=float(test_csv[test_csv['Patient']==id]['FVC'])
    
    week=int(test_csv[test_csv['Patient']==id]['Weeks'])
    
    predicted=float(submit[(submit['Patient']==id) & (submit['Weeks']==week) ]['FVC'])
    
    confidence_dict[id]=abs(real-predicted)
    


## === cell 50
confidence_dict


## === cell 51
confidence=[]
for i in range(len(submit)):
    confidence.append(confidence_dict[submit.iloc[i,0]])
submit['Confidence']=confidence


## === cell 52
submit['Patient']=submit['Patient']+'_'+(submit['Weeks'].astype(str))
submit.drop(['Weeks'],axis=1,inplace=True)
submit.columns=['Patient_Week','FVC','Confidence']


## === cell 53
submit.to_csv('submission.csv',index=False)


## --- ERROR in outputing the csv:
Invalid submission: Patient_Week ID00014637202177757139317_-12 in submission does not exist in answers
