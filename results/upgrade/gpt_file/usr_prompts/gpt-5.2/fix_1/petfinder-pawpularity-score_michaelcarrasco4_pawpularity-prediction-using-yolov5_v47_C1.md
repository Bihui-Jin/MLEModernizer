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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
tqdm==4.67.1

# 4. Data file paths

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

# 5. Target score

20.26344653457029

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math
from tqdm.notebook import tqdm
import imageio
import torch
import matplotlib.patches as patches
import os 
import matplotlib.image as img
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
digits = datasets.load_digits()
from sklearn.naive_bayes import BernoulliNB

## === cell 1
train = pd.read_csv('../input/petfinder-pawpularity-score/train.csv',nrows=10)
test = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')

## === cell 2
plt.figure(figsize= (15, 15))
sns.heatmap(train.corr(), annot=True, fmt='.1g' )
plt.title('Correlation Matrix', fontweight='bold', fontsize=20)
plt.show()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/459983744.py in <cell line: 0>()
      1 plt.figure(figsize= (15, 15))
----> 2 sns.heatmap(train.corr(), annot=True, fmt='.1g' )
      3 plt.title('Correlation Matrix', fontweight='bold', fontsize=20)
      4 plt.show()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: '1a8795e64a294ed0c95132e18ee198e1'

## === cell 3
train.hist(column='Pawpularity', bins=20)

## === cell 4
train.hist(column='Pawpularity',by='Blur', bins=10)
plt.xlabel('Blur')
plt.ylabel('Pawpularity')


## === cell 5
train.hist(column='Pawpularity',by='Near', bins=10)
plt.xlabel('Near')
plt.ylabel('Pawpularity')

## === cell 6
train.hist(column='Pawpularity',by='Group', bins=10)
plt.xlabel('Group')
plt.ylabel('Pawpularity')

## === cell 7
train.hist(column='Pawpularity',by='Near', bins=10)
plt.xlabel('Near')
plt.ylabel('Pawpularity')

## === cell 10
train_image='../input/petfinder-pawpularity-score/train'
top_images=train.sort_values(by='Pawpularity',ascending=False)
top_images=top_images['Id'][:6]

fig=plt.figure(figsize=(30,30))
fig.suptitle('Top 6 Pawpularity Score',fontsize=80)
for i in range(0,6):
    image=img.imread(os.path.join(train_image,list(top_images)[i]+'.jpg'))
    fi=fig.add_subplot(2,3,i+1)
    plt.imshow(image)  
    
plt.show()    



## === cell 11
worst_images=train.sort_values(by='Pawpularity',ascending=True)
worst_images=worst_images['Id'][:6]
fig=plt.figure(figsize=(30,30))
fig.suptitle('Bottom 6 Pawpularity Score',fontsize=80)
for i in range(0,6):
    image=img.imread(os.path.join(train_image,list(worst_images)[i]+'.jpg'))
    fi=fig.add_subplot(2,3,i+1)
    plt.imshow(image)  
    
plt.show()    

## === cell 13

!cp -R '../input/torch-hub/torch/root/.cache/torch' '/root/.cache/torch'

!cp -R '../input/torch-hub/ultralytics/root/.config/Ultralytics' '/root/.config/Ultralytics'
yolov5x6_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4068806281.py in <cell line: 0>()
      2 
      3 get_ipython().system("cp -R '../input/torch-hub/ultralytics/root/.config/Ultralytics' '/root/.config/Ultralytics'")
----> 4 yolov5x6_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

/usr/local/lib/python3.11/dist-packages/torch/hub.py in load(repo_or_dir, model, source, trust_repo, force_reload, verbose, skip_validation, *args, **kwargs)
    645         )
    646 
--> 647     model = _load_local(repo_or_dir, model, *args, **kwargs)
    648     return model
    649 

/usr/local/lib/python3.11/dist-packages/torch/hub.py in _load_local(hubconf_dir, model, *args, **kwargs)
    671     with _add_to_sys_path(hubconf_dir):
    672         hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
--> 673         hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
    674 
    675         entry = _load_entry_from_hubconf(hub_module, model)

/usr/local/lib/python3.11/dist-packages/torch/hub.py in _import_module(name, path)
    113     module = importlib.util.module_from_spec(spec)
    114     assert isinstance(spec.loader, Loader)
--> 115     spec.loader.exec_module(module)
    116     return module
    117 

/usr/lib/python3.11/importlib/_bootstrap_external.py in exec_module(self, module)

/usr/lib/python3.11/importlib/_bootstrap.py in _call_with_frames_removed(f, *args, **kwds)

~/.cache/torch/hub/ultralytics_yolov5_master/hubconf.py in <module>
     11 """
     12 
---> 13 from ultralytics.utils.patches import torch_load
     14 
     15 

ModuleNotFoundError: No module named 'ultralytics'

## === cell 14

def get_image_file_path(image_id):
    return f'../input/petfinder-pawpularity-score/train/{image_id}.jpg'


train['file_path'] = train['Id'].apply(get_image_file_path)

## === cell 15
widths = []
heights = []
ratios = []
for file_path in (train['file_path']):
    image = imageio.imread(file_path)
    h, w, _ = image.shape
    heights.append(h)
    widths.append(w)
    ratios.append(w / h)

## === cell 16
def get_image_info(file_path, plot=False):
    image = imageio.imread(file_path)
    h, w, c = image.shape
    
    if plot: # Debug Plots
        fig, ax = plt.subplots(1, 2, figsize=(8,8))
        ax[0].set_title('Pets detected in Image', size=16)
        ax[0].imshow(image)
        
    results = yolov5x6_model(image, augment=True)
    
    pet_pixels = np.zeros(shape=[h, w], dtype=np.uint8)
    
    h, w, _ = image.shape
    image_info = { 
        'n_pets': 0, # Number of pets in the image
        'labels': [], # Label assigned to found objects
        'thresholds': [], # confidence score
    }
    
    pets_found = []
    
    for x1, y1, x2, y2, treshold, label in results.xyxy[0].cpu().detach().numpy():
        label = results.names[int(label)]
        if label in ['dog', 'cat']:
            image_info['n_pets'] += 1
            image_info['labels'].append(label)
            image_info['thresholds'].append(treshold)

            
            pet_pixels[int(y1):int(y2), int(x1):int(x2)] = 1
            
            pets_found.append([x1, x2, y1, y2, label])

    if plot:
        for x1, x2, y1, y2, label in pets_found:
            c = 'red' if label == 'dog' else 'blue'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none')
            ax[0].add_patch(rect)
            ax[0].text(max(25, (x2+x1)/2), max(25, y1-h*0.02), label, c=c, ha='center', size=14)
                
    image_info['pet_ratio'] = pet_pixels.sum() / (h*w)

    if plot:
        ax[1].set_title('Pixels Containing Pets', size=16)
        ax[1].imshow(pet_pixels)
        plt.show()
        
    return image_info

## === cell 17
IMAGES_INFO = {
    'n_pets': [],
    'label': [],
    'pet_ratio': [],
}

## === cell 18
for file_path in train['file_path'].head(10):
    get_image_info(file_path, plot=True)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1059484184.py in <cell line: 0>()
      1 #Prints out results of yolov5 model
      2 for file_path in train['file_path'].head(10):
----> 3     get_image_info(file_path, plot=True)

/tmp/ipykernel_11/3877887181.py in get_image_info(file_path, plot)
     11 
     12     # Get YOLOV5 results using Test Time Augmentation for better result
---> 13     results = yolov5x6_model(image, augment=True)
     14 
     15     # Mask for pixels containing pets, initially all set to zero

NameError: name 'yolov5x6_model' is not defined

## === cell 19
for idx, file_path in enumerate(tqdm(train['file_path'])):
    image_info = get_image_info(file_path, plot=False)
    IMAGES_INFO['n_pets'].append(image_info['n_pets'])
    IMAGES_INFO['pet_ratio'].append(image_info['pet_ratio'])
    
    labels = image_info['labels']
    if len(set(labels)) == 1: # unanimous label
        IMAGES_INFO['label'].append(labels[0])
    elif len(set(labels)) > 1: # Get label with highest confidence
        IMAGES_INFO['label'].append(labels[0])
    else: # unknown label, yolo could not find pet
        IMAGES_INFO['label'].append('unknown')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/638218626.py in <cell line: 0>()
      1 for idx, file_path in enumerate(tqdm(train['file_path'])):
----> 2     image_info = get_image_info(file_path, plot=False)
      3     IMAGES_INFO['n_pets'].append(image_info['n_pets'])
      4     IMAGES_INFO['pet_ratio'].append(image_info['pet_ratio'])
      5 

/tmp/ipykernel_11/3877887181.py in get_image_info(file_path, plot)
     11 
     12     # Get YOLOV5 results using Test Time Augmentation for better result
---> 13     results = yolov5x6_model(image, augment=True)
     14 
     15     # Mask for pixels containing pets, initially all set to zero

NameError: name 'yolov5x6_model' is not defined

## === cell 20
for k, v in IMAGES_INFO.items():
    train[k] = v


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2198139048.py in <cell line: 0>()
      1 # Add Image Info to Train dataset
      2 for k, v in IMAGES_INFO.items():
----> 3     train[k] = v

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

ValueError: Length of values (0) does not match length of index (10)

## === cell 22
catc = sum(x == 'cat' for x in IMAGES_INFO['label'])
dogc = sum(x == 'dog' for x in IMAGES_INFO['label'])

## === cell 23
DOG_MEAN = train.loc[train['label'] == 'dog', 'Pawpularity'].mean()
CAT_MEAN = train.loc[train['label'] == 'cat', 'Pawpularity'].mean()
N_MEAN = train.loc[train['label'] == 'unknown', 'Pawpularity'].mean()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'label'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3678710693.py in <cell line: 0>()
----> 1 DOG_MEAN = train.loc[train['label'] == 'dog', 'Pawpularity'].mean()
      2 CAT_MEAN = train.loc[train['label'] == 'cat', 'Pawpularity'].mean()
      3 N_MEAN = train.loc[train['label'] == 'unknown', 'Pawpularity'].mean()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'label'

## === cell 24

fig = plt.figure()
ax = fig.add_axes([0,0,1,2])
langs = ['Dog', 'Cat','Unknown']
species = [DOG_MEAN,CAT_MEAN,N_MEAN]

ax.bar(langs,species)
ax.set_ylabel('Mean',fontsize=20)
ax.set_xlabel('Species',fontsize=20)
plt.show()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/181584960.py in <cell line: 0>()
      2 ax = fig.add_axes([0,0,1,2])
      3 langs = ['Dog', 'Cat','Unknown']
----> 4 species = [DOG_MEAN,CAT_MEAN,N_MEAN]
      5 
      6 ax.bar(langs,species)

NameError: name 'DOG_MEAN' is not defined

## === cell 25
plt.figure(figsize=(15, 8))
plt.title('Pawpularity Distribution', size=24)
train.loc[train['label'] != 2].groupby('label')['Pawpularity'].plot(kind='hist', 
                                                                            bins=20, alpha=0.50)
plt.legend(prop={'size': 20})


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'label'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/361108722.py in <cell line: 0>()
      1 plt.figure(figsize=(15, 8))
      2 plt.title('Pawpularity Distribution', size=24)
----> 3 train.loc[train['label'] != 2].groupby('label')['Pawpularity'].plot(kind='hist', 
      4                                                                             bins=20, alpha=0.50)
      5 plt.legend(prop={'size': 20})

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'label'

## === cell 26
train.label=train.label.replace('dog',0)
train.label=train.label.replace('cat',1)
train.label=train.label.replace('unknown',2)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3380046868.py in <cell line: 0>()
      1 #change label of of dog and cat to 0 and 1 respectively
----> 2 train.label=train.label.replace('dog',0)
      3 train.label=train.label.replace('cat',1)
      4 train.label=train.label.replace('unknown',2)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'label'

## === cell 28
train = pd.read_csv("../input/train-data/save_data.csv")


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2909553361.py in <cell line: 0>()
----> 1 train = pd.read_csv("../input/train-data/save_data.csv")
      2 #Skips yolo appending time with already saved data

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/train-data/save_data.csv'

## === cell 30
pd.set_option('display.max_columns', None)
train2=train.drop(columns=['file_path','Id'])
X = train2.drop(columns=['Pawpularity'])
y=train2['Pawpularity']

X_train, X_val, y_train, y_val =train_test_split(
    X, y, test_size=0.25, random_state=7)


## === cell 31

import matplotlib.patches as mpatches
def ActualvPredictionsGraph(y_test,y_pred,title):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.figure(figsize=(12,3))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.xlabel('Index ')
    plt.ylabel('Pawpularity ')
    plt.title(title,fontdict = {'fontsize' : 15})
    plt.legend(handles = [mpatches.Patch(color='red', label='prediction'),mpatches.Patch(color='blue', label='actual')])
    plt.show()
    return

## === cell 32

model_params = {
    'svc': {
        'model': svm.SVR(),
        'params' : {
            'C': [1,5,50,100],
            'kernel': ['rbf','linear','poly']
        }  
    },
    'Decision_tree': {
        'model': tree.DecisionTreeRegressor(),
        'params' : {
            'max_depth': [1,10,20,40,500,1500],
            'min_samples_split': [5,10,20,50],
            'min_samples_leaf' :[10,20,50],
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': np.logspace(0,-9, num=100)
        }
    

    },

}
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })   
results = pd.DataFrame(scores,columns=['model','best_score','best_params'])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/730517564.py in <cell line: 0>()
     29 for model_name, mp in model_params.items():
     30     clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
---> 31     clf.fit(X_train, y_train)
     32     scores.append({
     33         'model': model_name,

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
    872                 return results
    873 
--> 874             self._run_search(evaluate_candidates)
    875 
    876             # multimetric is determined here because in the case of a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
   1386     def _run_search(self, evaluate_candidates):
   1387         """Search all candidates in param_grid"""
-> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
   1389 
   1390 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
    831                         **fit_and_score_kwargs,
    832                     )
--> 833                     for (cand_idx, parameters), (split_idx, (train, test)) in product(
    834                         enumerate(candidate_params), enumerate(cv.split(X, y, groups))
    835                     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
    350             )
    351 
--> 352         for train, test in super().split(X, y, groups):
    353             yield train, test
    354 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
     83         X, y, groups = indexable(X, y, groups)
     84         indices = np.arange(_num_samples(X))
---> 85         for test_index in self._iter_test_masks(X, y, groups):
     86             train_index = indices[np.logical_not(test_index)]
     87             test_index = indices[test_index]

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _iter_test_masks(self, X, y, groups)
    731 
    732     def _iter_test_masks(self, X, y=None, groups=None):
--> 733         test_folds = self._make_test_folds(X, y)
    734         for i in range(self.n_splits):
    735             yield test_folds == i

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _make_test_folds(self, X, y)
    693         min_groups = np.min(y_counts)
    694         if np.all(self.n_splits > y_counts):
--> 695             raise ValueError(
    696                 "n_splits=%d cannot be greater than the"
    697                 " number of members in each class." % (self.n_splits)

ValueError: n_splits=3 cannot be greater than the number of members in each class.

## === cell 33
pd.options.display.max_columns = None
pd.options.display.max_rows = None
results



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2098689882.py in <cell line: 0>()
      1 pd.options.display.max_columns = None
      2 pd.options.display.max_rows = None
----> 3 results
      4 

NameError: name 'results' is not defined

## === cell 35
model=tree.DecisionTreeRegressor(splitter='best',max_depth= 3, min_samples_leaf= 20,min_samples_split=2,
                                max_features='auto')
tree_model=model.fit(X_train,y_train)
tree_y_pred = tree_model.predict(X_val)


r2=(r2_score(y_val,tree_y_pred))*100
print("R2 score:",r2)
MSE=mean_squared_error(y_val, tree_y_pred)
RMSE = math.sqrt(MSE)
x=RMSE
print("Root Mean Square Error:",RMSE)
ActualvPredictionsGraph(y_val[0:50], tree_y_pred[0:50], "Actual vs. Predicted over 50 values")
ActualvPredictionsGraph(y_val[0:9912], tree_y_pred[0:9912], "Actual vs. Predicted over all values")

## === cell 36
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model,max_depth=3, 
                   filled=True)

## === cell 37
model=GaussianNB(var_smoothing= 0.533669923120631)
gaussian_model=model.fit(X_train,y_train)
gaussian_y_pred = gaussian_model.predict(X_val)

r2=(r2_score(y_val,gaussian_y_pred))
print("R2 score:",r2)
MSE=mean_squared_error(y_val, gaussian_y_pred)
RMSE = math.sqrt(MSE)
y=RMSE
print("Root Mean Square Error:",RMSE)
ActualvPredictionsGraph(y_val[0:50], gaussian_y_pred[0:50], "Actual vs. Predicted over 50 values")
ActualvPredictionsGraph(y_val[0:9912], gaussian_y_pred[0:9912], "Actual vs. Predicted over all values")

## === cell 38
model=svm.SVR(C=10,kernel='linear')
svm_model=model.fit(X_train,y_train)
svm_y_pred = svm_model.predict(X_val)

r2=(r2_score(y_val,svm_y_pred))
print("R2 score:",r2)
MSE=mean_squared_error(y_val, svm_y_pred)
RMSE = math.sqrt(MSE)
z=RMSE
print("Root Mean Square Error:",RMSE)
ActualvPredictionsGraph(y_val[0:50], svm_y_pred[0:50], "Actual vs. Predicted over 50 values")
ActualvPredictionsGraph(y_val[0:9912], svm_y_pred[0:9912], "Actual vs. Predicted over all values")

## === cell 39
a=tree_y_pred
b=gaussian_y_pred
c=svm_y_pred
pred_final = (a+b+c)/3.0



r2=(r2_score(y_val,pred_final))
print("R2 score:",r2)
MSE=mean_squared_error(y_val, pred_final)
RMSE = math.sqrt(MSE)
z=RMSE
print("Root Mean Square Error:",RMSE)
ActualvPredictionsGraph(y_val[0:50], pred_final[0:50], "Actual vs. Predicted over 50 values")
ActualvPredictionsGraph(y_val[0:9912], pred_final[0:9912], "Actual vs. Predicted over all values")

## === cell 40
names=['SVM.SVR','Decision Tree','GaussianNB']
values=[z,x,y]
plt.title('Root Mean Squared Error')
plt.ylabel('RMSE Value')
plt.bar(names,values)

## === cell 41
test = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')


## === cell 42
def get_image_file_path(image_id):
    return f'../input/petfinder-pawpularity-score/test/{image_id}.jpg'

test['file_path'] = test['Id'].apply(get_image_file_path)

## === cell 43
widths = []
heights = []
ratios = []
for file_path in (test['file_path']):
    image = imageio.imread(file_path)
    h, w, _ = image.shape
    heights.append(h)
    widths.append(w)
    ratios.append(w / h)

## === cell 44
for file_path in test['file_path'].head(10):
    get_image_info(file_path, plot=True)

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1462311883.py in <cell line: 0>()
      1 for file_path in test['file_path'].head(10):
----> 2     get_image_info(file_path, plot=True)

/tmp/ipykernel_11/3877887181.py in get_image_info(file_path, plot)
     11 
     12     # Get YOLOV5 results using Test Time Augmentation for better result
---> 13     results = yolov5x6_model(image, augment=True)
     14 
     15     # Mask for pixels containing pets, initially all set to zero

NameError: name 'yolov5x6_model' is not defined

## === cell 45
IMAGES_INFO2 = {
    'n_pets': [],
    'label': [],
    'pet_ratio': [],
}


## === cell 46
for idx, file_path in enumerate(tqdm(test['file_path'])):
    image_info = get_image_info(file_path, plot=False)
    IMAGES_INFO2['n_pets'].append(image_info['n_pets'])
    IMAGES_INFO2['pet_ratio'].append(image_info['pet_ratio'])
    
    labels = image_info['labels']
    if len(set(labels)) == 1: # unanimous label
        IMAGES_INFO2['label'].append(labels[0])
    elif len(set(labels)) > 1: # Get label with highest confidence
        IMAGES_INFO2['label'].append(labels[0])
    else: # unknown label, yolo could not find pet
        IMAGES_INFO2['label'].append('unknown')

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3972568638.py in <cell line: 0>()
      1 for idx, file_path in enumerate(tqdm(test['file_path'])):
----> 2     image_info = get_image_info(file_path, plot=False)
      3     IMAGES_INFO2['n_pets'].append(image_info['n_pets'])
      4     IMAGES_INFO2['pet_ratio'].append(image_info['pet_ratio'])
      5 

/tmp/ipykernel_11/3877887181.py in get_image_info(file_path, plot)
     11 
     12     # Get YOLOV5 results using Test Time Augmentation for better result
---> 13     results = yolov5x6_model(image, augment=True)
     14 
     15     # Mask for pixels containing pets, initially all set to zero

NameError: name 'yolov5x6_model' is not defined

## === cell 47
for k, v in IMAGES_INFO2.items():
    test[k] = v
test=test.drop(columns=['Id','file_path'])

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/973977368.py in <cell line: 0>()
      1 for k, v in IMAGES_INFO2.items():
----> 2     test[k] = v
      3 test=test.drop(columns=['Id','file_path'])

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

ValueError: Length of values (0) does not match length of index (992)

## === cell 48
test.label=test.label.replace('dog',0)
test.label=test.label.replace('cat',1)
test.label=test.label.replace('unknown',2)
print(test)

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/276170786.py in <cell line: 0>()
      1 #change label of of and cat to 0 and 1 respectively
----> 2 test.label=test.label.replace('dog',0)
      3 test.label=test.label.replace('cat',1)
      4 test.label=test.label.replace('unknown',2)
      5 print(test)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'label'

## === cell 49
tree_y_pred = tree_model.predict(test)
svm_y_pred = svm_model.predict(test)
a=tree_y_pred
b=svm_y_pred
pred_final = (a+b)/2.0

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2512462543.py in <cell line: 0>()
----> 1 tree_y_pred = tree_model.predict(test)
      2 svm_y_pred = svm_model.predict(test)
      3 a=tree_y_pred
      4 b=svm_y_pred
      5 pred_final = (a+b)/2.0

/usr/local/lib/python3.11/dist-packages/sklearn/tree/_classes.py in predict(self, X, check_input)
    424         """
    425         check_is_fitted(self)
--> 426         X = self._validate_X_predict(X, check_input)
    427         proba = self.tree_.predict(X)
    428         n_samples = X.shape[0]

/usr/local/lib/python3.11/dist-packages/sklearn/tree/_classes.py in _validate_X_predict(self, X, check_input)
    390         """Validate the training data on predict (probabilities)."""
    391         if check_input:
--> 392             X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
    393             if issparse(X) and (
    394                 X.indices.dtype != np.intc or X.indptr.dtype != np.intc

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- Id
- file_path


## === cell 50
test = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')

test = pd.DataFrame(test, columns=['Id'])

## === cell 52
submission=pd.DataFrame(columns=['Id','Pawpularity'])
submission.head()
submission['Id'] = test.Id
submission['Pawpularity'] = pred_final
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2230861901.py in <cell line: 0>()
      2 submission.head()
      3 submission['Id'] = test.Id
----> 4 submission['Pawpularity'] = pred_final
      5 submission.to_csv('submission.csv', index=False)

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

ValueError: Length of values (3) does not match length of index (992)
