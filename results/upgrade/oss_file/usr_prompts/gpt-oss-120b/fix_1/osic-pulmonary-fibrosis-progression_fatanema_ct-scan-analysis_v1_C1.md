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

-6.972037427099178

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



## === cell 1
import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state



import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

import tables
import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import tqdm
from typing import Dict
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

from colorama import Fore, Back, Style


import pydicom

import warnings
warnings.filterwarnings('ignore')



import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg
from tabulate import tabulate
import missingno as msno 
from IPython.display import display_html
from PIL import Image
import gc
import cv2
from scipy.stats import pearsonr

import pydicom # for DICOM images
from skimage.transform import resize
import copy
import re

from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True) 

import warnings
warnings.filterwarnings("ignore")


custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']
sns.palplot(sns.color_palette(custom_colors))

from skimage.transform import resize

sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2

def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    
    img[img==max]=mean
    img[img==min]=mean
    
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image


    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0


    
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

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
    return mask*img

## === cell 3
patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/"
patient_names = os.listdir(patient_dir)

## === cell 4
patient_names.remove('ID00011637202177653955184')
patient_names.remove('ID00052637202186188008618')

## === cell 5

import h5py
filename = '../input/64-org/org_scans_64_v4.h5'


with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    data_sample = list(f[a_group_key])

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1208065815.py in <cell line: 0>()
      6 
      7 
----> 8 with h5py.File(filename, "r") as f:
      9     # List all groups
     10     print("Keys: %s" % f.keys())

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/64-org/org_scans_64_v4.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 7
pixels_mean = data_sample

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2065755448.py in <cell line: 0>()
----> 1 pixels_mean = data_sample

NameError: name 'data_sample' is not defined

## === cell 8
size_img = 64
df = pd.DataFrame([np.array(patient_names),np.array(pixels_mean).reshape(174,size_img*size_img)],['Patient','Pixels'])

df = df.T

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2192937450.py in <cell line: 0>()
      1 size_img = 64
----> 2 df = pd.DataFrame([np.array(patient_names),np.array(pixels_mean).reshape(174,size_img*size_img)],['Patient','Pixels'])
      3 #df_names = pd.DataFrame(data=np.array(patient_names).flatten())
      4 
      5 df = df.T

NameError: name 'pixels_mean' is not defined

## === cell 9
for i in range(len(df.Pixels)):
    df.Pixels[i] = df.Pixels[i]/max(df.Pixels[i])

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2462600384.py in <cell line: 0>()
----> 1 for i in range(len(df.Pixels)):
      2     df.Pixels[i] = df.Pixels[i]/max(df.Pixels[i])

NameError: name 'df' is not defined

## === cell 10
plt.figure()
plt.imshow(df.Pixels[0].reshape(64,64))
plt.show()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2749611224.py in <cell line: 0>()
      1 plt.figure()
----> 2 plt.imshow(df.Pixels[0].reshape(64,64))
      3 plt.show()

NameError: name 'df' is not defined

## === cell 11
plt.figure()
plt.imshow(df.Pixels[0].reshape(64,64))
plt.show()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2749611224.py in <cell line: 0>()
      1 plt.figure()
----> 2 plt.imshow(df.Pixels[0].reshape(64,64))
      3 plt.show()

NameError: name 'df' is not defined

## === cell 13
patient_dir_test = "../input/osic-pulmonary-fibrosis-progression/test/"
patient_names_test  = os.listdir(patient_dir_test )

## === cell 14
%%time

pixels1 = []
pixels_org = []

for i in range(0,len(patient_names_test)):
    datasets = []
    print(patient_names_test[i])

    files = []
    for dcm in list(os.listdir(patient_dir_test+patient_names_test[i])):
        files.append(dcm) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for dcm in files:
        path = patient_dir_test+patient_names_test[i] + "/" + dcm
        datasets.append(pydicom.dcmread(path))

    imgs = []
    for data in datasets:
        img = resize(data.pixel_array, (64, 64), anti_aliasing=True)
        imgs.append(img)
        
    pixels_org.append(np.mean(imgs,axis=0))

    pixels_ = []
    for i in range(0, len(imgs)):
        resized_img = resize(datasets[i-1].pixel_array, (64, 64), anti_aliasing=True)
        img = make_lungmask(resized_img)
        pixels_.append(img)
        
    pixels1.append(np.mean(pixels_,axis=0))
    if(i%10==0):
        print(i, ' Patients are analyzed!')

## === cell 15
pixels_mean_test = pixels_org

## === cell 16
size_img = 64

df_test = pd.DataFrame([np.array(patient_names_test),np.array(pixels_mean_test).reshape(len(patient_names_test),size_img*size_img)],['Patient','Pixels'])
df_test = df_test.T

for i in range(len(df_test)):
    df_test.Pixels[i] = df_test.Pixels[i].flatten()
    
for i in range(len(df_test.Pixels)):
    df_test.Pixels[i] = df_test.Pixels[i]/max(df_test.Pixels[i])
    

df_test2 = pd.DataFrame([np.array(patient_names_test),np.array(pixels_mean_test).reshape(len(patient_names_test),size_img*size_img)],['Patient','Pixels'])
df_test2 = df_test2.T

for i in range(len(df_test2)):
    df_test2.Pixels[i] = df_test2.Pixels[i].flatten()
    
for i in range(len(df_test2.Pixels)):
    df_test2.Pixels[i] = df_test2.Pixels[i]/max(df_test2.Pixels[i])



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2227202763.py in <cell line: 0>()
      1 size_img = 64
      2 
----> 3 df_test = pd.DataFrame([np.array(patient_names_test),np.array(pixels_mean_test).reshape(len(patient_names_test),size_img*size_img)],['Patient','Pixels'])
      4 df_test = df_test.T
      5 

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (19,) + inhomogeneous part.

## === cell 17
df.loc[174] = ['ID00011637202177653955184'] + [df.loc[df.Patient == 'ID00342637202287526592911'].Pixels.values[0]]
df.loc[175] = ['ID00052637202186188008618'] + [df.loc[df.Patient == 'ID00165637202237320314458'].Pixels.values[0]]

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1473725200.py in <cell line: 0>()
      1 #df.loc[173] = ['ID00105637202208831864134'] + [df.loc[df.Patient == 'ID00067637202189903532242'].Pixels.values[0]]
----> 2 df.loc[174] = ['ID00011637202177653955184'] + [df.loc[df.Patient == 'ID00342637202287526592911'].Pixels.values[0]]
      3 df.loc[175] = ['ID00052637202186188008618'] + [df.loc[df.Patient == 'ID00165637202237320314458'].Pixels.values[0]]

NameError: name 'df' is not defined

## === cell 18
np.shape(df)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4073285669.py in <cell line: 0>()
----> 1 np.shape(df)

NameError: name 'df' is not defined

## === cell 19
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)

## === cell 20
ROOT = "../input/osic-pulmonary-fibrosis-progression"
BATCH_SIZE= 128

## === cell 21
train = pd.read_csv(f"{ROOT}/train.csv")
train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
test = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(test.drop('Weeks', axis=1), on="Patient")

## === cell 22
df['WHERE'] = 'train'
df_test['WHERE'] = 'val'
df_test2['WHERE'] = 'test'

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/978549392.py in <cell line: 0>()
----> 1 df['WHERE'] = 'train'
      2 df_test['WHERE'] = 'val'
      3 df_test2['WHERE'] = 'test'

NameError: name 'df' is not defined

## === cell 23
ct_data = []
ct_data = df.append(df_test)
ct_data = ct_data.append(df_test2)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4194891053.py in <cell line: 0>()
      1 ct_data = []
----> 2 ct_data = df.append(df_test)
      3 ct_data = ct_data.append(df_test2)

NameError: name 'df' is not defined

## === cell 24
np.shape(ct_data)

## === cell 25
train['WHERE'] = 'train'
test['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = train.append([test, sub])

data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3908082426.py in <cell line: 0>()
      2 test['WHERE'] = 'val'
      3 sub['WHERE'] = 'test'
----> 4 data = train.append([test, sub])
      5 
      6 data['min_week'] = data['Weeks']

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 26
data['max_week'] = data['Weeks']
data.loc[data.WHERE=='test','max_week'] = np.nan
data['max_week'] = data.groupby('Patient')['max_week'].transform('max')

data['DLCO'] = data.FVC
data['min_FEV1'] = data.FVC

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    104         try:
--> 105             long_value = int(arg, 16)
    106             if long_value > 0xFFFFFFFF:

ValueError: invalid literal for int() with base 16: 'Weeks'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1046             try:
-> 1047                 tag = Tag(key)
   1048             except Exception as exc:

/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    117             if long_value is None:
--> 118                 raise ValueError(
    119                     f"Unable to create an element tag from '{arg}': "

ValueError: Unable to create an element tag from 'Weeks': unknown DICOM element keyword or an invalid int

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/231133199.py in <cell line: 0>()
----> 1 data['max_week'] = data['Weeks']
      2 data.loc[data.WHERE=='test','max_week'] = np.nan
      3 data['max_week'] = data.groupby('Patient')['max_week'].transform('max')
      4 
      5 data['DLCO'] = data.FVC

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1047                 tag = Tag(key)
   1048             except Exception as exc:
-> 1049                 raise KeyError(f"'{key}'") from exc
   1050 
   1051         elem = self._dict[tag]

KeyError: "'Weeks'"

## === cell 27
base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC','Percent','DLCO','min_FEV1']].copy()
base.columns = ['Patient','min_FVC','min_Percent','min_DLCO','min_FEV']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)
data = data.merge(base, on='Patient', how='left')


base1 = data.loc[data.Weeks == data.max_week]
base1 = base1[['Patient','FVC','Percent']].copy()
base1.columns = ['Patient','max_FVC','max_Percent']
base1['nb'] = 1
base1['nb'] = base1.groupby('Patient')['nb'].transform('cumsum')
base1 = base1[base1.nb==1]
base1.drop('nb', axis=1, inplace=True)
data = data.merge(base1, on='Patient', how='left')

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1173379856.py in <cell line: 0>()
----> 1 base = data.loc[data.Weeks == data.min_week]
      2 base = base[['Patient','FVC','Percent','DLCO','min_FEV1']].copy()
      3 base.columns = ['Patient','min_FVC','min_Percent','min_DLCO','min_FEV']
      4 base['nb'] = 1
      5 base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

AttributeError: 'FileDataset' object has no attribute 'loc'

## === cell 28
data = data.merge(ct_data, on=['Patient','WHERE'])

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4178221314.py in <cell line: 0>()
----> 1 data = data.merge(ct_data, on=['Patient','WHERE'])

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

AttributeError: 'FileDataset' object has no attribute 'merge'

## === cell 29
data['min_FVC1'] =  0.84 -0.3/data['min_FVC'] #0.84 -0.3/data['min_FVC']
data['FEV1'] =  0.84 -0.3/data['FVC'] #0.84 -0.3/data['min_FVC']
data['base_week'] = data['Weeks'] - data['min_week']
data['Neg_Age'] = -1/(data['Age'])

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    104         try:
--> 105             long_value = int(arg, 16)
    106             if long_value > 0xFFFFFFFF:

ValueError: invalid literal for int() with base 16: 'min_FVC'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1046             try:
-> 1047                 tag = Tag(key)
   1048             except Exception as exc:

/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    117             if long_value is None:
--> 118                 raise ValueError(
    119                     f"Unable to create an element tag from '{arg}': "

ValueError: Unable to create an element tag from 'min_FVC': unknown DICOM element keyword or an invalid int

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/10234549.py in <cell line: 0>()
----> 1 data['min_FVC1'] =  0.84 -0.3/data['min_FVC'] #0.84 -0.3/data['min_FVC']
      2 data['FEV1'] =  0.84 -0.3/data['FVC'] #0.84 -0.3/data['min_FVC']
      3 data['base_week'] = data['Weeks'] - data['min_week']
      4 data['Neg_Age'] = -1/(data['Age'])

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1047                 tag = Tag(key)
   1048             except Exception as exc:
-> 1049                 raise KeyError(f"'{key}'") from exc
   1050 
   1051         elem = self._dict[tag]

KeyError: "'min_FVC'"

## === cell 30
COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    104         try:
--> 105             long_value = int(arg, 16)
    106             if long_value > 0xFFFFFFFF:

ValueError: invalid literal for int() with base 16: 'Sex'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1046             try:
-> 1047                 tag = Tag(key)
   1048             except Exception as exc:

/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    117             if long_value is None:
--> 118                 raise ValueError(
    119                     f"Unable to create an element tag from '{arg}': "

ValueError: Unable to create an element tag from 'Sex': unknown DICOM element keyword or an invalid int

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3876623227.py in <cell line: 0>()
      2 FE = []
      3 for col in COLS:
----> 4     for mod in data[col].unique():
      5         FE.append(mod)
      6         data[mod] = (data[col] == mod).astype(int)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1047                 tag = Tag(key)
   1048             except Exception as exc:
-> 1049                 raise KeyError(f"'{key}'") from exc
   1050 
   1051         elem = self._dict[tag]

KeyError: "'Sex'"

## === cell 31
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


data['age'] = min_max_scaler.fit_transform(data['Age'].values.reshape(-1,1))
data['BASE'] = min_max_scaler.fit_transform(data['min_FVC'].values.reshape(-1,1))
data['week'] = min_max_scaler.fit_transform(data['base_week'].values.reshape(-1,1))
data['org_week'] = min_max_scaler.fit_transform(data['Weeks'].values.reshape(-1,1))
data['percent'] = min_max_scaler.fit_transform(data['Percent'].values.reshape(-1,1))
data['min_fev'] = min_max_scaler.fit_transform(data['min_FVC1'].values.reshape(-1,1))
data['neg_age'] = min_max_scaler.fit_transform(data['Neg_Age'].values.reshape(-1,1))


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    104         try:
--> 105             long_value = int(arg, 16)
    106             if long_value > 0xFFFFFFFF:

ValueError: invalid literal for int() with base 16: 'Age'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1046             try:
-> 1047                 tag = Tag(key)
   1048             except Exception as exc:

/usr/local/lib/python3.11/dist-packages/pydicom/tag.py in Tag(arg, arg2)
    117             if long_value is None:
--> 118                 raise ValueError(
    119                     f"Unable to create an element tag from '{arg}': "

ValueError: Unable to create an element tag from 'Age': unknown DICOM element keyword or an invalid int

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2840963126.py in <cell line: 0>()
      4 
      5 
----> 6 data['age'] = min_max_scaler.fit_transform(data['Age'].values.reshape(-1,1))
      7 data['BASE'] = min_max_scaler.fit_transform(data['min_FVC'].values.reshape(-1,1))
      8 data['week'] = min_max_scaler.fit_transform(data['base_week'].values.reshape(-1,1))

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getitem__(self, key)
   1047                 tag = Tag(key)
   1048             except Exception as exc:
-> 1049                 raise KeyError(f"'{key}'") from exc
   1050 
   1051         elem = self._dict[tag]

KeyError: "'Age'"

## === cell 32
FE = ['Ex-smoker','org_week','Never smoked','week','percent','Currently smokes','age','Female','Male','BASE','min_fev','neg_age'] 
FE2 = ['Pixels'] 

## === cell 33
train = data.loc[data.WHERE=='train']
test = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/722603693.py in <cell line: 0>()
----> 1 train = data.loc[data.WHERE=='train']
      2 test = data.loc[data.WHERE=='val']
      3 sub = data.loc[data.WHERE=='test']

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

AttributeError: 'FileDataset' object has no attribute 'loc'

## === cell 34
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import concatenate

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3067305259.py in <cell line: 0>()
      1 from keras.layers import Input, Dense, InputLayer, Flatten, Reshape, BatchNormalization
----> 2 from keras.layers.convolutional import Conv1D
      3 from keras.layers.convolutional import MaxPooling1D
      4 from tensorflow.keras.layers import concatenate

ModuleNotFoundError: No module named 'keras.layers.convolutional'

## === cell 35
y = train['FVC'].values
y = y.astype(float)
z1 = train[FE].values
ze1 = sub[FE].values

z = train[FE2].values
ze = sub[FE2].values

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3656952428.py in <cell line: 0>()
      1 y = train['FVC'].values
      2 y = y.astype(float)
----> 3 z1 = train[FE].values
      4 ze1 = sub[FE].values
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['Ex-smoker', 'org_week', 'Never smoked', 'week', 'percent',\n       'Currently smokes', 'age', 'Female', 'Male', 'BASE', 'min_fev',\n       'neg_age'],\n      dtype='object')] are in the [columns]"

## === cell 36
final_z = []

for j in range(len(z)):
    final_z.append(np.squeeze([ x for x in z[j] if not (x==i).all()]))
    
final_ze = []

for j in range(len(ze)):
    final_ze.append(np.squeeze([ x for x in ze[j] if not (x==i).all()]))
    
    
z2 = np.squeeze(final_z)
ze2 = np.squeeze(final_ze)

pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/279833224.py in <cell line: 0>()
      1 final_z = []
      2 
----> 3 for j in range(len(z)):
      4 #    print(j)
      5     final_z.append(np.squeeze([ x for x in z[j] if not (x==i).all()]))

NameError: name 'z' is not defined

## === cell 37
from keras import initializers

## === cell 38
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D


## === cell 39
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.80]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

def make_model():
    z1 = L.Input((len(FE),))
    z2 = L.Input((64*64,), name="Patient")

    init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform',seed=42)

    x = L.Dense(100,kernel_initializer = init,bias_initializer='zeros', activation="relu")(z1)
    x = L.Dense(100,kernel_initializer = init,bias_initializer='zeros', activation="relu")(x)

    
    x = M.Model(inputs=z1, outputs=x)
    
    
    y = Reshape((64*64, 1))(z2)
    
    shortcut = y
    
    y = Conv1D(kernel_initializer=init, activation='relu', 
                    padding="same", filters=4, kernel_size=8)(y)
    y = Conv1D(kernel_initializer=init, 
                    padding="same", filters=4, kernel_size=8)(y)

    y = Add()([y, shortcut])
    
    y= Activation('relu')(y)


    y = MaxPooling1D(pool_size=4)(y)

    y = Flatten()(y)

    
    y = L.Dense(256,kernel_initializer = init,bias_initializer='zeros', activation="relu")(y)
    y = L.Dense(128,kernel_initializer = init,bias_initializer='zeros', activation="relu")(y)   
    y = L.Dense(64,kernel_initializer = init,bias_initializer='zeros', activation="relu")(y)   
    y = L.Dense(32,kernel_initializer = init,bias_initializer='zeros', activation="relu")(y)   
    y = L.Dense(16,kernel_initializer = init,bias_initializer='zeros', activation="relu")(y)   
    y = L.Dense(8,kernel_initializer = init,bias_initializer='zeros', activation="relu")(y)

    y = M.Model(inputs=z2, outputs=y)


    combined = concatenate([x.output, y.output])




    
    p1 = L.Dense(3, activation="relu", name="p1")(combined)
    p2 = L.Dense(3, activation="linear", name="p2")(combined)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model([z1, z2], preds, name="CNN")
    model.compile(loss=mloss(0.80), optimizer=tf.keras.optimizers.Adam(lr=0.11, beta_1=0.92, beta_2=0.998, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model

## === cell 40
net = make_model()
print(net.summary())
print(net.count_params())

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4247974141.py in <cell line: 0>()
----> 1 net = make_model()
      2 print(net.summary())
      3 print(net.count_params())

/tmp/ipykernel_11/300935549.py in make_model()
     47     shortcut = y
     48 
---> 49     y = Conv1D(kernel_initializer=init, activation='relu', 
     50                     padding="same", filters=4, kernel_size=8)(y)
     51     y = Conv1D(kernel_initializer=init, 

NameError: name 'Conv1D' is not defined

## === cell 41
%%time

NFOLD = 5
kf = KFold(n_splits=NFOLD)
seed_everything(42)

pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))

results_t = []
results_v = []

cnt = 0
for tr_idx, val_idx in kf.split(z1):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model()
    net.fit([z1[tr_idx],z2[tr_idx]], y[tr_idx], batch_size=BATCH_SIZE, epochs=800, 
            validation_data=([z1[val_idx],z2[val_idx]], y[val_idx]), verbose=0)#,callbacks=[early_stopping]) #
    print("train", net.evaluate([z1[tr_idx],z2[tr_idx]], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate([z1[val_idx],z2[val_idx]], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    results_t.append(net.evaluate([z1[tr_idx],z2[tr_idx]], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    results_v.append(net.evaluate([z1[val_idx],z2[val_idx]], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")
    pred[val_idx] = net.predict([z1[val_idx],z2[val_idx]], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
net.fit([z1,z2], y, batch_size=BATCH_SIZE, epochs=800, verbose=0)#,callbacks=[early_stopping]) #
pe = net.predict([ze1,ze2], batch_size=BATCH_SIZE, verbose=0) #/ NFOLD

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'ze' is not defined

## === cell 42
print('training loss + score: ',np.mean(results_t,axis=0))
print('validation loss + score: ',np.mean(results_v,axis=0))
print('Final loss + score: ',np.mean(results_t,axis=0)*np.mean(results_v,axis=0))

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/11657606.py in <cell line: 0>()
----> 1 print('training loss + score: ',np.mean(results_t,axis=0))
      2 print('validation loss + score: ',np.mean(results_v,axis=0))
      3 print('Final loss + score: ',np.mean(results_t,axis=0)*np.mean(results_v,axis=0))

NameError: name 'results_t' is not defined

## === cell 45
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2785872833.py in <cell line: 0>()
----> 1 sigma_opt = mean_absolute_error(y, pred[:, 1])
      2 unc = pred[:,2] - pred[:, 0]
      3 sigma_mean = np.mean(unc)
      4 print(sigma_opt, sigma_mean)

NameError: name 'pred' is not defined

## === cell 46
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs],'ro', label="ground truth")
plt.plot(pred[idxs, 0],lw=2, label="q25")
plt.plot(pred[idxs, 1],lw=2, label="q50")
plt.plot(pred[idxs, 2],lw=2, label="q75")
plt.legend(loc="best")
plt.show()

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1460587059.py in <cell line: 0>()
      1 idxs = np.random.randint(0, y.shape[0], 100)
      2 plt.plot(y[idxs],'ro', label="ground truth")
----> 3 plt.plot(pred[idxs, 0],lw=2, label="q25")
      4 plt.plot(pred[idxs, 1],lw=2, label="q50")
      5 plt.plot(pred[idxs, 2],lw=2, label="q75")

NameError: name 'pred' is not defined

## === cell 47
print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2787804858.py in <cell line: 0>()
----> 1 print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())

NameError: name 'unc' is not defined

## === cell 48
plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2567398650.py in <cell line: 0>()
----> 1 plt.hist(unc)
      2 plt.title("uncertainty in prediction")
      3 plt.show()

NameError: name 'unc' is not defined

## === cell 49
sub['FVC1'] = 0.995  * pe[:, 1]
sub['Confidence1'] = pe[:, 2] - pe[:, 0]

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4110659920.py in <cell line: 0>()
----> 1 sub['FVC1'] = 0.995  * pe[:, 1]
      2 sub['Confidence1'] = pe[:, 2] - pe[:, 0]

NameError: name 'pe' is not defined

## === cell 50
subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/791857972.py in <cell line: 0>()
----> 1 subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC1', 'Confidence1'] not in index"

## === cell 51
subm.loc[~subm.FVC1.isnull()].head(10)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1757875118.py in <cell line: 0>()
----> 1 subm.loc[~subm.FVC1.isnull()].head(10)

NameError: name 'subm' is not defined

## === cell 52
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3734005870.py in <cell line: 0>()
----> 1 subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
      2 if sigma_mean<70:
      3     subm['Confidence'] = sigma_opt
      4 else:
      5     subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']

NameError: name 'subm' is not defined

## === cell 53
subm.describe().T

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/142471840.py in <cell line: 0>()
----> 1 subm.describe().T

NameError: name 'subm' is not defined

## === cell 54
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/32670954.py in <cell line: 0>()
      1 otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
      2 for i in range(len(otest)):
----> 3     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
      4     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

NameError: name 'subm' is not defined

## === cell 55
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1736978578.py in <cell line: 0>()
----> 1 subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

NameError: name 'subm' is not defined

## === cell 56
good =pd.read_csv("../input/best-aug30th/submission-14.csv")


plt.figure(figsize=(20,10))
plt.plot(range(len(good.FVC)), good.FVC,'ro',ms=2,label='Best')
plt.plot(range(len(subm.FVC)), subm.FVC,'bo',ms=2,label='Current')

plt.legend()
plt.show()

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2414499647.py in <cell line: 0>()
----> 1 good =pd.read_csv("../input/best-aug30th/submission-14.csv")
      2 
      3 
      4 plt.figure(figsize=(20,10))
      5 plt.plot(range(len(good.FVC)), good.FVC,'ro',ms=2,label='Best')

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/best-aug30th/submission-14.csv'

## === cell 57
plt.figure(figsize=(20,10))
plt.plot(range(len(good.Confidence)), good.Confidence,'ro',ms=2,label='Best')
plt.plot(range(len(subm.Confidence)), subm.Confidence,'bo',ms=2,label='Current')

plt.legend()
plt.show()

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/263249032.py in <cell line: 0>()
      1 plt.figure(figsize=(20,10))
----> 2 plt.plot(range(len(good.Confidence)), good.Confidence,'ro',ms=2,label='Best')
      3 plt.plot(range(len(subm.Confidence)), subm.Confidence,'bo',ms=2,label='Current')
      4 
      5 plt.legend()

NameError: name 'good' is not defined
