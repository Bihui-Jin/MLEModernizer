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

3.10

# 2. Installed packages

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
protobuf==6.33.0
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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
import cv2


## === cell 1
dat = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")


## === cell 2
y = dat["Pawpularity"].to_numpy()
y = y/100


## === cell 5
class data:
    def __init__(self,path,x=128,y=128,labels=None):
        self.x = x
        self.y = y
        self.labels = labels
        self.image_list = [t.split(".")[0] for t in os.listdir(path)] 
        self.path = path
        self.batch = 0
        
    def load_batch(self,batch_size=1,shuffle=False):
        if shuffle:
            b = self.batch
            batch_list = self.image_list[b*batch_size:(b+1)*batch_size]
            self.batch = b+1
            if self.batch>len(self.image_list)//batch_size:
                self.batch=0
        else:
            batch_list = sample(self.image_list,batch_size)
        images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(self.path+image+".jpg"),(128,128)),cv2.COLOR_BGR2RGB)/255 for image in batch_list])
        labels = self.labels.loc[batch_list].to_numpy()/100

        return images,labels

    
path = "../input/petfinder-pawpularity-score/train/"
labels = dat.set_index("Id")["Pawpularity"]
c = data(path,labels=labels)


## === cell 6
import os as _os
import sys as _sys
import importlib as _importlib
import subprocess as _subprocess

try:
    import google.protobuf as _pb

    _pb_ver = getattr(_pb, "__version__", "")
except Exception:
    _pb_ver = ""

if not _pb_ver or int(_pb_ver.split(".", 1)[0]) >= 6:
    _subprocess.check_call(
        [_sys.executable, "-m", "pip", "install", "-q", "protobuf>=5,<6"]
    )
    for _m in list(_sys.modules):
        if _m.startswith("google.protobuf"):
            del _sys.modules[_m]
    _importlib.invalidate_caches()

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *


## === cell 7
model = Sequential()
model.add(Input(shape=(128,128,3)))
model.add(Conv2D(10,5))
model.add(MaxPooling2D())
model.add(Conv2D(20,3))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation="relu"))
model.add(Dense(1,activation="relu"))


## === cell 8
model.summary()


## === cell 9
model.compile(optimizer=Adam(1e-3),loss="mae")#,metrics="mae")


## === cell 10
ids = dat["Id"].to_list()
train_ids = ids[:int(len(ids)*0.8)]
val_ids = ids[int(len(ids)*0.8):int(len(ids)*0.9)]
test_ids = ids[int(len(ids)*0.9):]


## === cell 11
class data:
    def __init__(self,path,ids,x=128,y=128,labels=None):
        self.x = x
        self.y = y
        self.labels = labels
        self.image_list = ids
        self.path = path
        self.batch = 0
        
    def load_batch(self,batch_size=1,shuffle=False):
        if shuffle:
            b = self.batch
            batch_list = self.image_list[b*batch_size:(b+1)*batch_size]
            self.batch = b+1
            if self.batch>len(self.image_list)//batch_size:
                self.batch=0
        else:
            batch_list = sample(self.image_list,batch_size)
        images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(self.path+image+".jpg"),(128,128)),cv2.COLOR_BGR2RGB)/255 for image in batch_list])
        labels = self.labels.loc[batch_list].to_numpy()/100

        return images,labels
    
    def loader(self,batch_size=1,shuffle=False):
        while True:
            x,y = self.load_batch(batch_size,shuffle)
            yield x,y
    
path = "../input/petfinder-pawpularity-score/train/"
labels = dat.set_index("Id")["Pawpularity"]
c = data(path,labels=labels,ids=ids)
c_train = data(path,labels=labels,ids=train_ids)
c_val = data(path,labels=labels,ids=val_ids)
c_test = data(path,labels=labels,ids=test_ids)


## === cell 13
model.fit(c.loader(batch_size=128), steps_per_epoch=100, epochs=2)
model.fit(c.loader(batch_size=32), steps_per_epoch=100, epochs=1)


## === cell 15
testpath = "../input/petfinder-pawpularity-score/test/"
ids = [t.split(".")[0] for t in os.listdir(testpath)]


## === cell 16
batch_size=200
y_pred=np.zeros((len(ids),1))
for i in range(0,len(ids)//batch_size):
    images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(testpath+image+".jpg"),(128,128)),cv2.COLOR_BGR2RGB)/255 for image in ids[batch_size*i:batch_size*(i+1)]])
    y_pred[i*batch_size:(i+1)*batch_size] = model.predict(images)*100


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31merror[0m                                     Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3725212633.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0my_pred[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m([0m[0mlen[0m[0;34m([0m[0mids[0m[0;34m)[0m[0;34m,[0m[0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m0[0m[0;34m,[0m[0mlen[0m[0;34m([0m[0mids[0m[0;34m)[0m[0;34m//[0m[0mbatch_size[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mimages[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0;34m[[0m[0mcv2[0m[0;34m.[0m[0mcvtColor[0m[0;34m([0m[0mcv2[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mcv2[0m[0;34m.[0m[0mimread[0m[0;34m([0m[0mtestpath[0m[0;34m+[0m[0mimage[0m[0;34m+[0m[0;34m".jpg"[0m[0;34m)[0m[0;34m,[0m[0;34m([0m[0;36m128[0m[0;34m,[0m[0;36m128[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m[0mcv2[0m[0;34m.[0m[0mCOLOR_BGR2RGB[0m[0;34m)[0m[0;34m/[0m[0;36m255[0m [0;32mfor[0m [0mimage[0m [0;32min[0m [0mids[0m[0;34m[[0m[0mbatch_size[0m[0;34m*[0m[0mi[0m[0;34m:[0m[0mbatch_size[0m[0;34m*[0m[0;34m([0m[0mi[0m[0;34m+[0m[0;36m1[0m[0;34m)[0m[0;34m][0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0my_pred[0m[0;34m[[0m[0mi[0m[0;34m*[0m[0mbatch_size[0m[0;34m:[0m[0;34m([0m[0mi[0m[0;34m+[0m[0;36m1[0m[0;34m)[0m[0;34m*[0m[0mbatch_size[0m[0;34m][0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mimages[0m[0;34m)[0m[0;34m*[0m[0;36m100[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3725212633.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m      2[0m [0my_pred[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m([0m[0mlen[0m[0;34m([0m[0mids[0m[0;34m)[0m[0;34m,[0m[0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m0[0m[0;34m,[0m[0mlen[0m[0;34m([0m[0mids[0m[0;34m)[0m[0;34m//[0m[0mbatch_size[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mimages[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0;34m[[0m[0mcv2[0m[0;34m.[0m[0mcvtColor[0m[0;34m([0m[0mcv2[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mcv2[0m[0;34m.[0m[0mimread[0m[0;34m([0m[0mtestpath[0m[0;34m+[0m[0mimage[0m[0;34m+[0m[0;34m".jpg"[0m[0;34m)[0m[0;34m,[0m[0;34m([0m[0;36m128[0m[0;34m,[0m[0;36m128[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m[0mcv2[0m[0;34m.[0m[0mCOLOR_BGR2RGB[0m[0;34m)[0m[0;34m/[0m[0;36m255[0m [0;32mfor[0m [0mimage[0m [0;32min[0m [0mids[0m[0;34m[[0m[0mbatch_size[0m[0;34m*[0m[0mi[0m[0;34m:[0m[0mbatch_size[0m[0;34m*[0m[0;34m([0m[0mi[0m[0;34m+[0m[0;36m1[0m[0;34m)[0m[0;34m][0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0my_pred[0m[0;34m[[0m[0mi[0m[0;34m*[0m[0mbatch_size[0m[0;34m:[0m[0;34m([0m[0mi[0m[0;34m+[0m[0;36m1[0m[0;34m)[0m[0;34m*[0m[0mbatch_size[0m[0;34m][0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mimages[0m[0;34m)[0m[0;34m*[0m[0;36m100[0m[0;34m[0m[0;34m[0m[0m

[0;31merror[0m: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 18
round2 = lambda x,y=None:round(x+1e-15,y)
y_pred = [round2(t[0],2) for t in y_pred]
