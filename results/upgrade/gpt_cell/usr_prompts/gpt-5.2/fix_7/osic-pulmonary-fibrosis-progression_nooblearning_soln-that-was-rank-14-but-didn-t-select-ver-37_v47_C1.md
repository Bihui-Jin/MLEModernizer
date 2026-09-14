# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import importlib
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _pb_ver is not None and _major(_pb_ver) is not None and _major(_pb_ver) >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    importlib.invalidate_caches()
    for _m in list(sys.modules.keys()):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from skimage import morphology
from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import tensorflow.keras.backend as k


## === cell 1
train_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')


## === cell 2
train_csv


## === cell 3
base_week=train_csv.groupby('Patient')['Weeks'].min()
base_week_list=[]
for i in range(len(train_csv)):
    base_week_list.append(base_week[train_csv.iloc[i,0]])
train_csv['base_week']=base_week_list


base_week=train_csv.groupby('Patient')['Weeks'].min()
count_from_base_week=[]
for i in range(len(train_csv)):
    count_from_base_week.append(train_csv.iloc[i,1]-base_week[train_csv.iloc[i,0]])
train_csv['count_from_base_week']=count_from_base_week

confidence=np.zeros(train_csv.shape[0])
train_csv['confidence']=confidence

base_fvc_dict={}
for id in train_csv['Patient'].unique():
    base_fvc_dict[id]=np.array(train_csv[(train_csv['Patient']==id) & (train_csv['Weeks']==base_week[id])]['FVC'])[0]
base_fvc=[]
for i in range(len(train_csv)):
    base_fvc.append(base_fvc_dict[train_csv.iloc[i,0]])
train_csv['base_fvc']=base_fvc

base_fev1_dict={}
for id in train_csv['Patient'].unique():
    A=train_csv[train_csv['Patient']==id]["base_fvc"].unique()[0]
    B=train_csv[train_csv['Patient']==id]["Age"].unique()[0]
    if train_csv[train_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_fev1_dict[id]=0.77*A+0.32+0.0069*B
    else:
        base_fev1_dict[id]=0.77*A+0.28+0.0052*B
        
base_fev1=[]
for i in range(len(train_csv)):
    base_fev1.append(base_fev1_dict[train_csv.iloc[i,0]])
train_csv['base_fev1']=base_fev1

base_week_percent_dict={}
for id in train_csv['Patient'].unique():
    base_week_percent_dict[id]=np.array(train_csv[(train_csv['Patient']==id) & (train_csv['Weeks']==base_week[id])]['Percent'])[0]
    
base_week_percent=[]
for i in range(len(train_csv)):
    base_week_percent.append(base_week_percent_dict[train_csv.iloc[i,0]])
train_csv['base_week_percent']=base_week_percent

train_csv['base fev1/base fvc']=train_csv['base_fev1']/train_csv['base_fvc']

train_csv['base_height']=(train_csv['base_fvc']+9030)/77.0


base_weight_dict={}
for id in train_csv['Patient'].unique():
    FVC=train_csv[train_csv['Patient']==id]["base_fvc"].unique()[0]
    A=train_csv[train_csv['Patient']==id]["Age"].unique()[0]
    H=train_csv[train_csv['Patient']==id]["base_height"].unique()[0]
    if train_csv[train_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_weight_dict[id]=(FVC+5458-49*H+8*A)/12.0
    else:
        base_weight_dict[id]=(FVC+3863-37*H+6*A)/14.0
base_weight=[]
for i in range(len(train_csv)):
    base_weight.append(base_weight_dict[train_csv.iloc[i,0]])
train_csv['base_weight']=base_weight

train_csv['base_bmi']=train_csv['base_weight']/((train_csv['base_height']/100)**2)


## === cell 4
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()#sex
train_csv.iloc[:,5]=lb.fit_transform(train_csv.iloc[:,5])
lb2=LabelEncoder()#ss
train_csv.iloc[:,6]=lb2.fit_transform(train_csv.iloc[:,6])


## === cell 5
from sklearn.preprocessing import OneHotEncoder
oh1=OneHotEncoder(handle_unknown='ignore')
smoke_cat=pd.DataFrame(oh1.fit_transform(train_csv[['SmokingStatus']]).toarray(),columns=['smoking cat 0','smoking cat 1','smoking cat 2'])
train_csv=pd.concat([train_csv,smoke_cat],axis=1)


## === cell 6
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_scaled=pd.DataFrame(sc.fit_transform(train_csv[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])
train_scaled['Sex']=train_csv['Sex']
train_scaled['smoking cat 0']=train_csv['smoking cat 0']
train_scaled['smoking cat 1']=train_csv['smoking cat 1']


## === cell 7
train_scaled


## === cell 8
train_csv


## === cell 9
sub=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 10
test_week=[]
patient_id=[]
for i in range(len(sub)):
    test_week.append(int(sub.iloc[i,0].split('_')[-1]))
    patient_id.append(sub.iloc[i,0].split('_')[0])
sub['Patient']=patient_id
sub['Weeks']=test_week
sub.drop(['FVC','Confidence'],axis=1,inplace=True)

base_fvc=test_csv.groupby('Patient')['FVC'].min()
fvc=[]
for i in range(len(sub)):
    fvc.append(base_fvc[sub.iloc[i,1]])
sub['base_fvc']=fvc

base_fev1_dict_test={}
for id in sub['Patient'].unique():
    A=sub[sub['Patient']==id]["base_fvc"].unique()[0]
    B=test_csv[test_csv['Patient']==id]["Age"].unique()[0]
    if test_csv[test_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_fev1_dict_test[id]=0.77*A+0.32+0.0069*B
    else:
        base_fev1_dict_test[id]=0.77*A+0.28+0.0052*B
        
base_fev1_test=[]
for i in range(len(sub)):
    base_fev1_test.append(base_fev1_dict_test[sub.iloc[i,1]])
sub['base_fev1']=base_fev1_test


sub['base_height']=(sub['base_fvc']+9030)/77.0



base_weight_dict_test={}
for id in sub['Patient'].unique():
    FVC=sub[sub['Patient']==id]["base_fvc"].unique()[0]
    A=test_csv[test_csv['Patient']==id]["Age"].unique()[0]
    H=sub[sub['Patient']==id]["base_height"].unique()[0]
    if test_csv[test_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_weight_dict_test[id]=(FVC+5458-49*H+8*A)/12.0
    else:
        base_weight_dict_test[id]=(FVC+3863-37*H+6*A)/14.0
base_weight_test=[]
for i in range(len(sub)):
    base_weight_test.append(base_weight_dict_test[sub.iloc[i,1]])
sub['base_weight']=base_weight_test


test_csv.iloc[:,5]=lb.transform(test_csv.iloc[:,5])
test_csv.iloc[:,6]=lb2.transform(test_csv.iloc[:,6])

percent_dict={}
for id in test_csv['Patient'].unique():
    percent_dict[id]=float(test_csv[test_csv['Patient']==id]['Percent'])
    
sex_dict={}
for id in test_csv['Patient'].unique():
    sex_dict[id]=int(test_csv[test_csv['Patient']==id]['Sex'])

age_dict={}
for id in test_csv['Patient'].unique():
    age_dict[id]=int(test_csv[test_csv['Patient']==id]['Age'])
    
ss_dict={}
for id in test_csv['Patient'].unique():
    ss_dict[id]=int(test_csv[test_csv['Patient']==id]['SmokingStatus'])

percent=[]
sex=[]
age=[]
ss=[]
for i in range(len(sub)):
    percent.append(percent_dict[sub.iloc[i,1]])
    sex.append(sex_dict[sub.iloc[i,1]])
    age.append(age_dict[sub.iloc[i,1]])
    ss.append(ss_dict[sub.iloc[i,1]])    
sub['base_week_percent']=percent
sub['Age']=age
sub['Sex']=sex
sub['SmokingStatus']=ss


base_week_test=test_csv.groupby('Patient')['Weeks'].min()
count_from_base_week_test=[]
base_week=[]
for i in range(len(sub)):
    count_from_base_week_test.append(sub.iloc[i,2]-base_week_test[sub.iloc[i,1]])
    base_week.append(base_week_test[sub.iloc[i,1]])
sub['count_from_base_week']=count_from_base_week_test
sub['base_week']=base_week


sub['base fev1/base fvc']=sub['base_fev1']/sub['base_fvc']

sub['base_bmi']=sub['base_weight']/((sub['base_height']/100.0)**2)


## === cell 11
smoke_num = pd.to_numeric(sub["SmokingStatus"], errors="coerce")

fill_value = int(lb2.transform([lb2.classes_[0]])[0])
smoke_num = smoke_num.fillna(fill_value).astype(int)

smoke_str = pd.Series(lb2.inverse_transform(smoke_num.values), index=sub.index)

smoke_cat_test = pd.DataFrame(
    oh1.transform(pd.DataFrame({"SmokingStatus": smoke_str})).toarray(),
    columns=["smoking cat 0", "smoking cat 1", "smoking cat 2"],
)
sub = pd.concat([sub, smoke_cat_test], axis=1)


## === cell 12
sub_scaled=pd.DataFrame(sc.transform(sub[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])
sub_scaled['Sex']=sub['Sex']
sub_scaled['smoking cat 0']=sub['smoking cat 0']
sub_scaled['smoking cat 1']=sub['smoking cat 1']


## === cell 13
sub_scaled


## === cell 14
x=np.array(train_scaled[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])
y=np.array(train_csv[['FVC','confidence']])

from sklearn.model_selection import train_test_split
xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)


## === cell 15
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
    
    
def model_loss(ytrue,ypred):   # this loss penalises both prediction and confidence value
    eps=1.0
    
    fvc_pred=ypred[:,0]
    sigmas=ypred[:,1]+eps     # so as to avoid log(0) . these are predicted connfidences
    
    ans=tf.math.log(sigmas)
    ans=ans+((ytrue[:,0]-fvc_pred)**2)/(2*sigmas**2)
    
    return tf.reduce_mean(ans)


## === cell 16
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    
    sigma = y_pred[:,1]
    fvc_pred = y_pred[:,0]
    

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:,0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return tf.reduce_mean(metric)

def huber_loss(y_true, y_pred):
    
    error = y_true[:,0] - y_pred[:,0]
    is_small_error = tf.abs(error) <= 1000.0
    quad_loss = tf.square(error) / 2
    linear_loss = 1000*tf.abs(error) - tf.square(C2)*0.5
    return tf.reduce_mean(tf.where(is_small_error, quad_loss, linear_loss))


def custom_loss(y_true, y_pred):
    return huber_loss(y_true, y_pred) + score(y_true, y_pred)


## === cell 17
lr_scheduler=tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,monitor='val_loss',mode='min',patience=150,verbose=0)
class best_weights(tf.keras.callbacks.Callback):
    def __init__(self):
        self.metric_op=-30.0
        self.weights_op=None
        self.epoch_op=-1
    def on_epoch_end(self,epoch,logs={}):
        if logs['val_metric']>=self.metric_op:
            self.metric_op=logs['val_metric']
            self.epoch_op=epoch
            self.weights_op=self.model.get_weights()
    def on_train_end(self,logs={}):
        self.model.set_weights(self.weights_op)
        print('BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}'.format(self.epoch_op+1,self.metric_op))
        
        

class metrics_call(tf.keras.callbacks.Callback):
    def __init__(self,mertic,xtrain,ytrain,xvalid,yvalid):
        self.metric=metric
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xvalid=xvalid
        self.yvalid=yvalid
        
    def on_epoch_end(self,epoch,logs={}):
        train_preds=self.model.predict(self.xtrain)
        val_preds=self.model.predict(self.xvalid)
        logs['val_metric']=self.metric(self.yvalid[:,0],val_preds[:,0],val_preds[:,1])
        
        

def run_model(xtrain,ytrain,xvalid,yvalid,epoch=50):
    input=tf.keras.layers.Input(shape=xtrain.shape[1:])
    noisy=tf.keras.layers.GaussianNoise(0.4)(input)
    
    d1=tf.keras.layers.Dense(128,activation='relu')(noisy)
    d2=tf.keras.layers.Dense(128,activation='relu')(d1)
    d3=tf.keras.layers.Dense(128,activation='relu')(d2)
    mean_out1=tf.keras.layers.Dense(1)(d3)
    std_den1=tf.keras.layers.Dense(1)(d3)
    
    d4=tf.keras.layers.Dense(128,activation='relu')(noisy)
    d5=tf.keras.layers.Dense(128,activation='relu')(d4)
    d6=tf.keras.layers.Dense(128,activation='relu')(d5)
    mean_out2=tf.keras.layers.Dense(1)(d6)
    std_den2=tf.keras.layers.Dense(1)(d6)
    
    d7=tf.keras.layers.Dense(128,activation='relu')(noisy)
    d8=tf.keras.layers.Dense(128,activation='relu')(d7)
    d9=tf.keras.layers.Dense(128,activation='relu')(d8)
    mean_out3=tf.keras.layers.Dense(1)(d9)
    std_den3=tf.keras.layers.Dense(1)(d9)
    
    mean_combine=tf.keras.layers.Concatenate()([mean_out1,mean_out2,mean_out3])
    std_combine=tf.keras.layers.Concatenate()([std_den1,std_den2,std_den3])
    mean_final=tf.keras.layers.Dense(1)(mean_combine)
    std_final_den=tf.keras.layers.Dense(1)(std_combine)
    std_final=tf.keras.layers.Lambda(lambda x: tf.abs(x))(std_final_den)
    output=tf.keras.layers.Concatenate()([mean_final,std_final])
    model=tf.keras.models.Model(inputs=input,outputs=output)
    
    model.compile(loss=lambda ytrue,ypred: custom_loss(ytrue,ypred),optimizer=tf.keras.optimizers.Adam(lr=0.001))
    print(model.summary())
    history=model.fit(xtrain,ytrain,epochs=epoch,batch_size=256,validation_data=(xvalid,yvalid),verbose=0,callbacks=[metrics_call(metric,xtrain,ytrain,xvalid,yvalid),best_weights()])
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.ylim(-10,10)
    plt.grid(True)

    return model


## === cell 18
def _run_model_patched(xtrain, ytrain, xvalid, yvalid, epoch=50):
    input = tf.keras.layers.Input(shape=xtrain.shape[1:])
    noisy = tf.keras.layers.GaussianNoise(0.4)(input)

    d1 = tf.keras.layers.Dense(128, activation="relu")(noisy)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(128, activation="relu")(d2)
    mean_out1 = tf.keras.layers.Dense(1)(d3)
    std_den1 = tf.keras.layers.Dense(1)(d3)

    d4 = tf.keras.layers.Dense(128, activation="relu")(noisy)
    d5 = tf.keras.layers.Dense(128, activation="relu")(d4)
    d6 = tf.keras.layers.Dense(128, activation="relu")(d5)
    mean_out2 = tf.keras.layers.Dense(1)(d6)
    std_den2 = tf.keras.layers.Dense(1)(d6)

    d7 = tf.keras.layers.Dense(128, activation="relu")(noisy)
    d8 = tf.keras.layers.Dense(128, activation="relu")(d7)
    d9 = tf.keras.layers.Dense(128, activation="relu")(d8)
    mean_out3 = tf.keras.layers.Dense(1)(d9)
    std_den3 = tf.keras.layers.Dense(1)(d9)

    mean_combine = tf.keras.layers.Concatenate()([mean_out1, mean_out2, mean_out3])
    std_combine = tf.keras.layers.Concatenate()([std_den1, std_den2, std_den3])
    mean_final = tf.keras.layers.Dense(1)(mean_combine)
    std_final_den = tf.keras.layers.Dense(1)(std_combine)
    std_final = tf.keras.layers.Lambda(lambda x: tf.abs(x))(std_final_den)
    output = tf.keras.layers.Concatenate()([mean_final, std_final])
    model = tf.keras.models.Model(inputs=input, outputs=output)

    model.compile(
        loss=lambda ytrue, ypred: custom_loss(ytrue, ypred),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    print(model.summary())
    history = model.fit(
        xtrain,
        ytrain,
        epochs=epoch,
        batch_size=256,
        validation_data=(xvalid, yvalid),
        verbose=0,
        callbacks=[
            metrics_call(metric, xtrain, ytrain, xvalid, yvalid),
            best_weights(),
        ],
    )
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.ylim(-10, 10)
    plt.grid(True)

    return model


run_model = _run_model_patched

model = run_model(xtrain, ytrain, xvalid, yvalid, epoch=500)


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4065829183.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     57[0m [0mrun_model[0m [0;34m=[0m [0m_run_model_patched[0m[0;34m[0m[0;34m[0m[0m
[1;32m     58[0m [0;34m[0m[0m
[0;32m---> 59[0;31m [0mmodel[0m [0;34m=[0m [0mrun_model[0m[0;34m([0m[0mxtrain[0m[0;34m,[0m [0mytrain[0m[0;34m,[0m [0mxvalid[0m[0;34m,[0m [0myvalid[0m[0;34m,[0m [0mepoch[0m[0;34m=[0m[0;36m500[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/4065829183.py[0m in [0;36m_run_model_patched[0;34m(xtrain, ytrain, xvalid, yvalid, epoch)[0m
[1;32m     35[0m     )
[1;32m     36[0m     [0mprint[0m[0;34m([0m[0mmodel[0m[0;34m.[0m[0msummary[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 37[0;31m     history = model.fit(
[0m[1;32m     38[0m         [0mxtrain[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     39[0m         [0mytrain[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optree/ops.py[0m in [0;36mtree_map[0;34m(func, tree, is_leaf, none_is_leaf, namespace, *rests)[0m
[1;32m    764[0m     [0mleaves[0m[0;34m,[0m [0mtreespec[0m [0;34m=[0m [0m_C[0m[0;34m.[0m[0mflatten[0m[0;34m([0m[0mtree[0m[0;34m,[0m [0mis_leaf[0m[0;34m,[0m [0mnone_is_leaf[0m[0;34m,[0m [0mnamespace[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    765[0m     [0mflat_args[0m [0;34m=[0m [0;34m[[0m[0mleaves[0m[0;34m][0m [0;34m+[0m [0;34m[[0m[0mtreespec[0m[0;34m.[0m[0mflatten_up_to[0m[0;34m([0m[0mr[0m[0;34m)[0m [0;32mfor[0m [0mr[0m [0;32min[0m [0mrests[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 766[0;31m     [0;32mreturn[0m [0mtreespec[0m[0;34m.[0m[0munflatten[0m[0;34m([0m[0mmap[0m[0;34m([0m[0mfunc[0m[0;34m,[0m [0;34m*[0m[0mflat_args[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    767[0m [0;34m[0m[0m
[1;32m    768[0m [0;34m[0m[0m

[0;31mValueError[0m: Invalid dtype: object

## === cell 19
xtest=np.array(sub_scaled[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])

yans=model.predict(xtest)

sub['FVC']=yans[:,0]
sub['Confidence']=yans[:,1]
sub.drop(['Patient','Weeks','Age','Sex','SmokingStatus','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1','smoking cat 2'],axis=1,inplace=True)
