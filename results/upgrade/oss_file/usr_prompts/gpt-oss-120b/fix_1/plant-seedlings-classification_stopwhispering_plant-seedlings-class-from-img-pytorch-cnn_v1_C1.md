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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
google-ai-generativelanguage==0.6.15
google-api-core==2.28.1
google-api-python-client==2.177.0
google-auth==2.38.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.2
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pydata-google-auth==1.9.1
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.84508

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {DEVICE}')

if 'google.colab' in str(get_ipython()):
    NUM_EPOCHS = 50
    !pip install torchviz
    BASE_PATH = './drive/MyDrive/Colab/data/'
    from google.colab import drive
    drive.mount('/content/drive')
    
elif get_ipython().config.IPKernelApp.connection_file.startswith('/root/.local/share'):
    NUM_EPOCHS = 15
    BASE_PATH = '/kaggle/input/'
    !pip install torchviz
    
elif 'SHLVL' in os.environ:
    NUM_EPOCHS = 50
    BASE_PATH = '/kaggle/input/'
    !pip install torchviz

else:
    BASE_PATH = '../data/'
    NUM_EPOCHS = 2


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3833231356.py in <cell line: 0>()
     13 
     14 # running interactively in kaggle
---> 15 elif get_ipython().config.IPKernelApp.connection_file.startswith('/root/.local/share'):
     16     NUM_EPOCHS = 15
     17     BASE_PATH = '/kaggle/input/'

AttributeError: 'LazyConfigValue' object has no attribute 'startswith'

## === cell 1
import random
from tqdm.auto import tqdm
import numpy as np
from collections.abc import Callable
import locale
locale.setlocale(locale.LC_ALL, locale='')  # for thousands separator via ... print(f'{value:n}')"
import math
from itertools import islice
from collections.abc import Iterable, Generator
from pprint import pprint
import pathlib

from IPython.display import HTML, Image
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision import utils
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torchviz import make_dot
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sklearn.metrics
import cv2

my_seed = 123
random.seed(my_seed)
torch.manual_seed(my_seed)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3902648970.py in <cell line: 0>()
     26 from torch.utils.data import DataLoader
     27 from torch.nn.modules.loss import _Loss
---> 28 from torchviz import make_dot
     29 import torch.nn.functional as F
     30 from torch.utils.data import DataLoader, TensorDataset

ModuleNotFoundError: No module named 'torchviz'

## === cell 2
path_train = pathlib.Path(BASE_PATH + "plant-seedlings-classification/train/")
for x in path_train.iterdir():
    print(x)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/857477490.py in <cell line: 0>()
----> 1 path_train = pathlib.Path(BASE_PATH + "plant-seedlings-classification/train/")
      2 for x in path_train.iterdir():
      3     print(x)

NameError: name 'BASE_PATH' is not defined

## === cell 3
labels = [d.name for d in path_train.iterdir() if d.is_dir()]

labels_arr = []
path_arr = []

for label in labels:
    path_plant_dir = path_train.joinpath(label)
    print(f'{label}: {len(list(path_plant_dir.iterdir()))}')
    image_paths = list(path_plant_dir.iterdir())
    labels_arr.extend([label]*len(image_paths))
    path_arr.extend(image_paths)

df_meta = pd.DataFrame({'path': path_arr,
                         'label': labels_arr})
df_meta


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4064992815.py in <cell line: 0>()
      1 # collect labels and each label's image paths
----> 2 labels = [d.name for d in path_train.iterdir() if d.is_dir()]
      3 
      4 labels_arr = []
      5 path_arr = []

NameError: name 'path_train' is not defined

## === cell 4
fig, axes = plt.subplots(nrows=5,
                         ncols=4,
                         figsize=(15,15),
                        )

for i in range(20):
    image_id = random.randrange(len(df_meta))
    path = df_meta.iloc[image_id]['path']
    example_image = torchvision.io.read_image(str(path))  # [3, e.g. 196, e.g. 196], torch.uint8
    example_image = example_image.permute(1, 2, 0)  # [196, 196, 3]

    ax = axes[i//4, i%4]
    ax.imshow(X=example_image)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_title(df_meta.iloc[image_id]['label'])


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1868975559.py in <cell line: 0>()
      5 
      6 for i in range(20):
----> 7     image_id = random.randrange(len(df_meta))
      8     path = df_meta.iloc[image_id]['path']
      9     example_image = torchvision.io.read_image(str(path))  # [3, e.g. 196, e.g. 196], torch.uint8

NameError: name 'df_meta' is not defined

## === cell 5
shapes = np.zeros(shape=(len(df_meta),2), 
                  dtype=np.uint16)

for i, image_path in enumerate(df_meta['path']):
    image_path = df_meta.iloc[i]['path']
    image = cv2.imread(str(image_path),  # returns np.array of differentshape and dtype uint8
                       flags=cv2.IMREAD_COLOR)  # convert to 3 channel BGR (Blue-Green-Red)
    
    shapes[i] = image.shape[:2]
    
df_meta['width'] =  shapes[:, 0]
df_meta['height'] = shapes[:, 1]


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/245035104.py in <cell line: 0>()
      1 # collect image sizes
----> 2 shapes = np.zeros(shape=(len(df_meta),2), 
      3                   dtype=np.uint16)
      4 
      5 for i, image_path in enumerate(df_meta['path']):

NameError: name 'df_meta' is not defined

## === cell 6
df_meta.describe()  # statistical analysis on numerical cols


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1830808985.py in <cell line: 0>()
----> 1 df_meta.describe()  # statistical analysis on numerical cols

NameError: name 'df_meta' is not defined

## === cell 7
df_meta[df_meta['width'] == 3457]


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1410533536.py in <cell line: 0>()
      1 # Largest Image
----> 2 df_meta[df_meta['width'] == 3457]

NameError: name 'df_meta' is not defined

## === cell 8
df_meta['aspect_ratio'] = df_meta['width'] / df_meta['height']
print(min_ar := min(df_meta['aspect_ratio']))
print(max_ar := max(df_meta['aspect_ratio']))


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2172060258.py in <cell line: 0>()
      1 # relationship between width and height (how 'unsquare' are our images?)
----> 2 df_meta['aspect_ratio'] = df_meta['width'] / df_meta['height']
      3 print(min_ar := min(df_meta['aspect_ratio']))
      4 print(max_ar := max(df_meta['aspect_ratio']))

NameError: name 'df_meta' is not defined

## === cell 9
df_meta[df_meta['aspect_ratio'] < 0.9 ]


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3201152603.py in <cell line: 0>()
----> 1 df_meta[df_meta['aspect_ratio'] < 0.9 ]

NameError: name 'df_meta' is not defined

## === cell 10
df_meta[df_meta['aspect_ratio'] > 1.1 ]


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2449667841.py in <cell line: 0>()
----> 1 df_meta[df_meta['aspect_ratio'] > 1.1 ]

NameError: name 'df_meta' is not defined

## === cell 11
SIZE = 70

image_list: list[np.array] = []

for i, image_path in enumerate(df_meta['path']):
    image = cv2.imread(str(image_path),  # returns np.array of differentshape and dtype uint8
                   flags=cv2.IMREAD_COLOR)  # convert to 3 channel BGR (Blue-Green-Red)
    image_resized = cv2.resize(src=image,  # (70, 70, 3), uint8 0..255
                               dsize=(SIZE, SIZE))
    image_list.append(image_resized)

images = np.asarray(image_list)  # (4750, 70, 70, 3)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/228094358.py in <cell line: 0>()
      3 image_list: list[np.array] = []
      4 
----> 5 for i, image_path in enumerate(df_meta['path']):
      6     image = cv2.imread(str(image_path),  # returns np.array of differentshape and dtype uint8
      7                    flags=cv2.IMREAD_COLOR)  # convert to 3 channel BGR (Blue-Green-Red)

NameError: name 'df_meta' is not defined

## === cell 12
print(f'Memory Consumption of resized images: {images.nbytes :n}')


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1800257676.py in <cell line: 0>()
----> 1 print(f'Memory Consumption of resized images: {images.nbytes :n}')

NameError: name 'images' is not defined

## === cell 13
def convert_image_to_hsv(image):
    image_hsv = cv2.cvtColor(src=image,
                             code=cv2.COLOR_BGR2HSV)
    return image_hsv
    

def create_mask_for_plant(image_hsv):
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(src=image_hsv, 
                       lowerb=lower_hsv, 
                       upperb=upper_hsv)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                       ksize=(11,11))
    mask = cv2.morphologyEx(src=mask, 
                            op=cv2.MORPH_CLOSE, 
                            kernel=kernel)

    return mask

def mask_plant(image, mask):
    output = cv2.bitwise_and(src1=image, 
                             src2=image, 
                             mask=mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(src=image, 
                                     ksize=(0, 0), 
                                     sigmaX=3)
    image_sharp = cv2.addWeighted(src1=image, 
                                  alpha=1.5, 
                                  src2=image_blurred, 
                                  beta=-0.5, 
                                  gamma=0)
    return image_sharp


## === cell 14
random_indexes = [random.randint(0, len(images)) for _ in range(5)]
random_images = images[random_indexes]  # (5, 70, 70, 3)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1476530329.py in <cell line: 0>()
----> 1 random_indexes = [random.randint(0, len(images)) for _ in range(5)]
      2 random_images = images[random_indexes]  # (5, 70, 70, 3)

/tmp/ipykernel_11/1476530329.py in <listcomp>(.0)
----> 1 random_indexes = [random.randint(0, len(images)) for _ in range(5)]
      2 random_images = images[random_indexes]  # (5, 70, 70, 3)

NameError: name 'images' is not defined

## === cell 15
image_path = df_meta[df_meta['label'] == 'Small-flowered Cranesbill'].iloc[197]['path']
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # (e.g. 760, e.g. 760, 3)  # uint8

image_hsv = convert_image_to_hsv(image)
image_mask = create_mask_for_plant(image_hsv)
image_masked = mask_plant(image, image_mask)
image_sharpen = sharpen_image(image_masked)

fig, axs = plt.subplots(1, 5, figsize=(20, 20))
axs[0].imshow(image)
axs[1].imshow(image_hsv)
axs[2].imshow(image_mask)
axs[3].imshow(image_masked)
axs[4].imshow(image_sharpen)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3965733446.py in <cell line: 0>()
----> 1 image_path = df_meta[df_meta['label'] == 'Small-flowered Cranesbill'].iloc[197]['path']
      2 image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # (e.g. 760, e.g. 760, 3)  # uint8
      3 
      4 image_hsv = convert_image_to_hsv(image)
      5 image_mask = create_mask_for_plant(image_hsv)

NameError: name 'df_meta' is not defined

## === cell 16
fig, axes = plt.subplots(nrows=len(random_images), 
                        ncols=5, 
                        figsize=(20, 20))

for i, image in enumerate(random_images):

    image_hsv = convert_image_to_hsv(image)
    image_mask = create_mask_for_plant(image_hsv)
    image_masked = mask_plant(image, image_mask)
    image_sharpened = sharpen_image(image_masked)
    
    axes[i, 0].imshow(image)
    axes[i, 1].imshow(image_hsv)
    axes[i, 2].imshow(image_mask)
    axes[i, 3].imshow(image_masked)
    axes[i, 4].imshow(image_sharpened)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])


axes[0, 0].set_title('Original', fontsize=30)
axes[0, 1].set_title('HSV', fontsize=30)
axes[0, 2].set_title('Mask', fontsize=30)
axes[0, 3].set_title('Masked', fontsize=30)
axes[0, 4].set_title('Sharpened', fontsize=30)

fig.tight_layout()
plt.show()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1079070568.py in <cell line: 0>()
----> 1 fig, axes = plt.subplots(nrows=len(random_images), 
      2                         ncols=5,
      3                         figsize=(20, 20))
      4 
      5 for i, image in enumerate(random_images):

NameError: name 'random_images' is not defined

## === cell 17
masked_image_list = []

for image in images:
    image_hsv = convert_image_to_hsv(image)
    image_mask = create_mask_for_plant(image_hsv)
    image_masked = mask_plant(image, image_mask)
    image_sharpened = sharpen_image(image_masked)
    masked_image_list.append(image_sharpened)

masked_images = np.asarray(masked_image_list)  # (4750, 70, 70, 3)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2308382602.py in <cell line: 0>()
      1 masked_image_list = []
      2 
----> 3 for image in images:
      4     image_hsv = convert_image_to_hsv(image)
      5     image_mask = create_mask_for_plant(image_hsv)

NameError: name 'images' is not defined

## === cell 18
normalized_images = masked_images / 255  # uint8 -> float64

x = normalized_images


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2365064309.py in <cell line: 0>()
----> 1 normalized_images = masked_images / 255  # uint8 -> float64
      2 
      3 # we'll use these images for training
      4 x = normalized_images

NameError: name 'masked_images' is not defined

## === cell 19
df_meta['label'].value_counts().plot(kind='bar')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1434818851.py in <cell line: 0>()
----> 1 df_meta['label'].value_counts().plot(kind='bar')

NameError: name 'df_meta' is not defined

## === cell 20
label_encoder = sklearn.preprocessing.LabelEncoder()

y = label_encoder.fit_transform(df_meta['label'])  # np.array (4750,), int32, 0..11

example = ['Maize']
print(f'Example: {example} -> {(enc := label_encoder.transform(example))} -> {label_encoder.inverse_transform(enc)}')


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4185636645.py in <cell line: 0>()
----> 1 label_encoder = sklearn.preprocessing.LabelEncoder()
      2 
      3 y = label_encoder.fit_transform(df_meta['label'])  # np.array (4750,), int32, 0..11
      4 
      5 example = ['Maize']

NameError: name 'sklearn' is not defined

## === cell 21
assert len(x) == len(y)

randomized_indexes = np.random.permutation(len(x))  # (4750,)
x_rnd = x[randomized_indexes]  # still (4750, 70, 70, 3)
y_rnd = y[randomized_indexes]  # still (4750,)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2716147528.py in <cell line: 0>()
----> 1 assert len(x) == len(y)
      2 
      3 # we need to randomize x and y together
      4 randomized_indexes = np.random.permutation(len(x))  # (4750,)
      5 x_rnd = x[randomized_indexes]  # still (4750, 70, 70, 3)

NameError: name 'x' is not defined

## === cell 22
num_train = int(len(x) * 0.85)

x_train_arr = x_rnd[:num_train]  # (4037, 70, 70, 3), float64
x_val_arr = x_rnd[num_train:]  # (713, 70, 70, 3)

y_train_arr = y_rnd[:num_train]  # (4037,)
y_val_arr = y_rnd[num_train:]  # (713,)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3583102562.py in <cell line: 0>()
----> 1 num_train = int(len(x) * 0.85)
      2 
      3 x_train_arr = x_rnd[:num_train]  # (4037, 70, 70, 3), float64
      4 x_val_arr = x_rnd[num_train:]  # (713, 70, 70, 3)
      5 

NameError: name 'x' is not defined

## === cell 23
x_train = torch.tensor(x_train_arr.astype(np.float32)).to(DEVICE)  # [4037, 70, 70, 3], torch.float32
x_val = torch.tensor(x_val_arr.astype(np.float32)).to(DEVICE)  # [713, 70, 70, 3], torch.float32

y_train = torch.tensor(y_train_arr.astype(np.int64)).to(DEVICE)  # [4037], torch.int64
y_val = torch.tensor(y_val_arr.astype(np.int64)).to(DEVICE)  # [713], torch.int32


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3514825386.py in <cell line: 0>()
      1 # tensorize arrays
----> 2 x_train = torch.tensor(x_train_arr.astype(np.float32)).to(DEVICE)  # [4037, 70, 70, 3], torch.float32
      3 x_val = torch.tensor(x_val_arr.astype(np.float32)).to(DEVICE)  # [713, 70, 70, 3], torch.float32
      4 
      5 y_train = torch.tensor(y_train_arr.astype(np.int64)).to(DEVICE)  # [4037], torch.int64

NameError: name 'x_train_arr' is not defined

## === cell 24
BATCH_SIZE = 16

train_dataset = torch.utils.data.TensorDataset(x_train, 
                                               y_train)

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3533546430.py in <cell line: 0>()
      1 BATCH_SIZE = 16
      2 
----> 3 train_dataset = torch.utils.data.TensorDataset(x_train, 
      4                                                y_train)
      5 

NameError: name 'x_train' is not defined

## === cell 25
class CNNClassifier(torch.nn.Module):

    def __init__(self, 
                 dropout_probability=0.3,
                 num_labels=12):
        super(CNNClassifier, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,  # Number of channels in the input image
                            out_channels=32,  # Number of channels produced by the convolution
                            kernel_size=3, #  Size of the convolving kernel
                            stride=1,  # Stride of the convolution. Default: 1
                            padding=1,  # Padding added to all four sides of the input. Default: 0
                           ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, # the size of the window to take a max over
                               stride=2,  # the stride of the window. Default value is kernel_size
                              ),
            torch.nn.Dropout(p=dropout_probability,  # probability of an element to be zeroed. Default: 0.5
                            ),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,
                               stride=2),
            torch.nn.Dropout(p=dropout_probability))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, 
                               stride=2, 
                               padding=1),  # default: 0
            torch.nn.Dropout(p=dropout_probability)
            )
        
        self.flatten = torch.nn.Flatten()  # for feed-forward 

        
        self.fc1 = torch.nn.Linear(in_features=9 * 9 * 128,
                                   out_features=625,
                                   bias=True)
        
        self.fc2 = torch.nn.Linear(in_features=625,
                                   out_features=num_labels,
                                   bias=True)
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)  # initialize weights (seems to make no difference)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [batch_size, 70, 70, 3], torch.float32
        
        x = x.permute(dims=(0, 3, 1, 2))
        
        output_layer_1 = self.layer1(x)   # [batch_size, 32, 35, 35]
        output_layer_2 = self.layer2(output_layer_1)  # [batch_size, 64, 17, 17
        output_layer_3 = self.layer3(output_layer_2)  # [batch_size, 128, 9, 9]
        flattened = self.flatten(output_layer_3)  # [batch_size, 10368]
        
        output_fully_connected_1 = self.fc1(flattened)  # [batch_size, 625]
        output_fully_connected_2 = self.fc2(output_fully_connected_1)  # [batch_size, 12]

        return output_fully_connected_2


## === cell 26
c_temp = CNNClassifier().to(DEVICE)
x_batch, _ = next(iter(train_loader)) 
predictions = c_temp(x_batch)
make_dot(predictions)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1029454670.py in <cell line: 0>()
      2 c_temp = CNNClassifier().to(DEVICE)
      3 # to visualize with torchviz, we need some input that can pass through the model's forward() method.
----> 4 x_batch, _ = next(iter(train_loader))
      5 predictions = c_temp(x_batch)
      6 make_dot(predictions)

NameError: name 'train_loader' is not defined

## === cell 27
LEARNING_RATE = 0.001

classifier = CNNClassifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(),
                             lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  # reduce learning rate when model stops improving on validation dataset 
                                                       mode='min', 
                                                       verbose=True)


## === cell 28
def compute_metrics(classifier: CNNClassifier, 
                    loss_fn: Callable,
                    x: torch.Tensor, 
                    y: torch.Tensor
                   )->tuple[float, float, float]:
    
        y_pred_logits = classifier(x)
        loss = loss_fn(y_pred_logits, y).item()
    
        y_pred = y_pred_logits.argmax(dim=1)
        correct = (y_pred == y).type(torch.FloatTensor)
        accuracy = correct.mean().item()

        f1_score = sklearn.metrics.f1_score(y_true=y.cpu(), 
                                            y_pred=y_pred.cpu(),
                                            average='micro')  # multi-class problem
        
        return loss, accuracy, f1_score


## === cell 29
df_metrics = pd.DataFrame(columns=['loss_train', 'accuracy_train', 'f1_train', 
                                   'loss_val', 'accuracy_val', 'f1_val'],
                          index=range(NUM_EPOCHS),
                          dtype=float)

for epoch in tqdm(range(NUM_EPOCHS)):

    for batch, (x_train_batch, y_train_batch) in enumerate(train_loader):


        classifier.train()

        pred_train_batch_logits = classifier(x_train_batch)  # [batch_size, 12], float32

        optimizer.zero_grad()
        loss = loss_fn(pred_train_batch_logits,
                       y_train_batch)  # [], .item() is e.g. 2.291177988052368

        loss.backward()
        optimizer.step()
        
    classifier.eval()
    with torch.no_grad():
        loss_train, accuracy_train, f1_score_train = compute_metrics(classifier, loss_fn, x_train, y_train)
        loss_val, accuracy_val, f1_score_val = compute_metrics(classifier, loss_fn, x_val, y_val)

        df_metrics.iloc[epoch] = [loss_train, accuracy_train, f1_score_train,
                                  loss_val, accuracy_val, f1_score_val]
        
    scheduler.step(loss_val)
    print(f'Accuracy Validation after epoch {epoch}: {accuracy_val :.4f}  '
          f'(Train: {accuracy_train :.4f}) '
          f'LR = {optimizer.param_groups[0]["lr"]}\n')


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3720057421.py in <cell line: 0>()
      1 df_metrics = pd.DataFrame(columns=['loss_train', 'accuracy_train', 'f1_train', 
      2                                    'loss_val', 'accuracy_val', 'f1_val'],
----> 3                           index=range(NUM_EPOCHS),
      4                           dtype=float)
      5 

NameError: name 'NUM_EPOCHS' is not defined

## === cell 30
df_metrics.style.background_gradient(cmap='Blues')


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3835654791.py in <cell line: 0>()
----> 1 df_metrics.style.background_gradient(cmap='Blues')

NameError: name 'df_metrics' is not defined

## === cell 31
epochs = range(NUM_EPOCHS)

fig, ((ax1, ax2), (ax3, _)) = plt.subplots(nrows=2,
                                       ncols=2,
                                       figsize=(15,5),
                                          sharex=True)

ax1.plot(epochs, df_metrics['loss_train'], label='Training Loss')
ax1.plot(epochs, df_metrics['loss_val'], label='val Loss')
ax1.set_ylabel('Loss')
ax1.legend(loc='best')

ax2.plot(epochs, df_metrics['accuracy_train'], label='Training Accuracy')
ax2.plot(epochs, df_metrics['accuracy_val'], label='val Accuracy')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='best')

ax3.plot(epochs, df_metrics['f1_train'], label='Training F1-Score')
ax3.plot(epochs, df_metrics['f1_val'], label='val F1-Score')
ax3.set_ylabel('F1-Score')
ax3.legend(loc='best')
ax3.set_xlabel('Epochs')
ax3.set_xticks(np.arange(0, 
                         NUM_EPOCHS))

plt.suptitle('Training and Validation Metrics')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, 
                     NUM_EPOCHS))

plt.show()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/525799839.py in <cell line: 0>()
----> 1 epochs = range(NUM_EPOCHS)
      2 
      3 fig, ((ax1, ax2), (ax3, _)) = plt.subplots(nrows=2,
      4                                        ncols=2,
      5                                        figsize=(15,5),

NameError: name 'NUM_EPOCHS' is not defined

## === cell 32
path_test = pathlib.Path(BASE_PATH + "plant-seedlings-classification/test/")
image_list_test = []
filenames = []

for path in path_test.iterdir():
    image = cv2.imread(str(path),
                       flags=cv2.IMREAD_COLOR)
    image_resized = cv2.resize(src=image,
                               dsize=(SIZE, SIZE))
    image_hsv = convert_image_to_hsv(image_resized)
    image_mask = create_mask_for_plant(image_hsv)
    image_masked = mask_plant(image_resized, image_mask)
    image_sharpened = sharpen_image(image_masked)
    
    image_list_test.append(image_sharpened)
    
    filenames.append(path.name)

images_test = np.asarray(image_list_test)  # (794, 70, 70, 3)

normalized_images_test = images_test / 255


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2593849867.py in <cell line: 0>()
      1 # Load and preprocess test data
----> 2 path_test = pathlib.Path(BASE_PATH + "plant-seedlings-classification/test/")
      3 image_list_test = []
      4 filenames = []
      5 

NameError: name 'BASE_PATH' is not defined

## === cell 33

x_test = torch.tensor(normalized_images_test.astype(np.float32)).to(DEVICE)  # [794, 70, 70, 3], torch.float32

classifier.eval()
with torch.no_grad():
    y_pred_logits = classifier(x_test)  # [794, 12], torch.float32
    y_pred = y_pred_logits.argmax(dim=1)  # [794], torch.int64
    predicted_labels = y_pred.cpu().numpy()  # np.array (794,), int64
    
predicted_plants = label_encoder.inverse_transform(predicted_labels)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1676667410.py in <cell line: 0>()
      1 # Predict the labels
      2 
----> 3 x_test = torch.tensor(normalized_images_test.astype(np.float32)).to(DEVICE)  # [794, 70, 70, 3], torch.float32
      4 
      5 classifier.eval()

NameError: name 'normalized_images_test' is not defined

## === cell 34
pd.Series(predicted_plants).value_counts().plot(kind='bar')


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3162411822.py in <cell line: 0>()
----> 1 pd.Series(predicted_plants).value_counts().plot(kind='bar')

NameError: name 'predicted_plants' is not defined

## === cell 35
with open(BASE_PATH + "plant-seedlings-classification/sample_submission.csv")as f:
    print(f.readline())
    print(f.readline())
    print(f.readline())
    print('...')


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/709296096.py in <cell line: 0>()
      1 # this is how the submission file must look like
----> 2 with open(BASE_PATH + "plant-seedlings-classification/sample_submission.csv")as f:
      3     print(f.readline())
      4     print(f.readline())
      5     print(f.readline())

NameError: name 'BASE_PATH' is not defined

## === cell 36
df_submission = pd.DataFrame({'file': filenames,
                              'species': predicted_plants})
df_submission.to_csv('submission.csv',
                      index=False)

with open("submission.csv")as f:
    print(f.readline())
    print(f.readline())
    print(f.readline())
    print('...')


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3599716069.py in <cell line: 0>()
----> 1 df_submission = pd.DataFrame({'file': filenames,
      2                               'species': predicted_plants})
      3 df_submission.to_csv('submission.csv',
      4                       index=False)
      5 

NameError: name 'filenames' is not defined
