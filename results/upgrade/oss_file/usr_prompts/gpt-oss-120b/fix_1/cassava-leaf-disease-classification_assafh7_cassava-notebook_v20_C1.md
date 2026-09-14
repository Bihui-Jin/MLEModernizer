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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.837413115744938

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import seaborn as sns
import os, cv2, json
import matplotlib.pyplot as plt
import albumentations as aug
from keras.optimizers import SGD


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import warnings
warnings.simplefilter("ignore")

## === cell 3
general_path = '../input/cassava-leaf-disease-classification/'
os.listdir(general_path)

## === cell 4
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k) : v for k, v in map_classes.items()}
    
    print(json.dumps(map_classes, indent=4))

## === cell 5
input_files = os.listdir(os.path.join(general_path, "train_images"))
print(f"Number of train images: {len(input_files)}")

## === cell 6
img_shapes = {}
for image_name in os.listdir(os.path.join(general_path, "train_images"))[:300]:
    image = cv2.imread(os.path.join(general_path, "train_images", image_name))
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)

## === cell 7
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))

df_train["class_name"] = df_train["label"].map(map_classes)

df_train

## === cell 8
plt.figure(figsize=(8, 4))
sns.countplot(y="class_name", data=df_train);

## === cell 9
def visualize_batch(image_ids, labels,class_name):
    plt.figure(figsize=(16, 12))
    
    for ind, (image_id, label,class_name) in enumerate(zip(image_ids, labels,class_name)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(os.path.join(general_path, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Class {label}:  {class_name}", fontsize=12)
        plt.axis("off")
    
    plt.show()

## === cell 10
tmp_df = df_train[df_train["label"] == 0]
print(f"Total train images for class 0: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values

visualize_batch(image_ids, labels,class_name)

## === cell 11
tmp_df = df_train[df_train["label"] == 1]
print(f"Total train images for class 1: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values


visualize_batch(image_ids, labels,class_name)

## === cell 12
tmp_df = df_train[df_train["label"] == 2]
print(f"Total train images for class 2: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values


visualize_batch(image_ids,labels, class_name)

## === cell 13
tmp_df = df_train[df_train["label"] == 3]
print(f"Total train images for class 2: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values
visualize_batch(image_ids, labels,class_name)

## === cell 14
tmp_df = df_train[df_train["label"] == 4]
print(f"Total train images for class 4: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values

visualize_batch(image_ids, labels,class_name)

## === cell 15
def plot_augmentation(image_id, transform):
    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(general_path, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("original")
    
    plt.subplot(2, 2, 2)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.title("Augmentation -1")

    
    plt.subplot(2, 2, 3)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.title("Augmentation -2")
    
    plt.subplot(2, 2, 4)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.title("Augmentation -3")
    
    plt.show()

## === cell 16
transform_shift_scale_rotate = aug.ShiftScaleRotate(
    p=1.0, 
    shift_limit=(-0.3, 0.3), 
    scale_limit=(-0.1, 0.1), 
    rotate_limit=(-180, 180), 
    interpolation=0, 
    border_mode=4, 
)

plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)

## === cell 17
transform_coarse_dropout = aug.CoarseDropout(
    p=1.0, 
    max_holes=100, 
    max_height=50, 
    max_width=50, 
    min_holes=30, 
    min_height=20, 
    min_width=20,
)

plot_augmentation("1003442061.jpg", transform_coarse_dropout)

## === cell 18
transform_coarse_dropout = aug.HueSaturationValue(
    hue_shift_limit=0,
    sat_shift_limit=(40,80),
    val_shift_limit=(40,80)
)

plot_augmentation("5912799.jpg", transform_coarse_dropout)

## === cell 19
transform_coarse_dropout = aug.CLAHE(
 
     always_apply=False,
      p=1.0,
      clip_limit=(10, 30),
      tile_grid_size=(10, 10))
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 20
transform_coarse_dropout = aug.RandomFog( p=1.0)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 21
transform_coarse_dropout = aug.RandomSunFlare( p=1.0)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 22
transform_coarse_dropout = aug.RandomBrightness( p=1.0)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/958931524.py in <cell line: 0>()
----> 1 transform_coarse_dropout = aug.RandomBrightness( p=1.0)
      2 plot_augmentation("999329392.jpg", transform_coarse_dropout)

AttributeError: module 'albumentations' has no attribute 'RandomBrightness'

## === cell 23
transform_coarse_dropout = aug.RandomCrop(p=1,height = 512, width = 512)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 24
transform_coarse_dropout = aug.RGBShift(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 25
transform_coarse_dropout = aug.RandomSnow(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 26
transform_coarse_dropout = aug.HorizontalFlip(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 27
transform_coarse_dropout = aug.VerticalFlip(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 28
transform_coarse_dropout = aug.RandomContrast(limit = 0.5,p = 1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4102694802.py in <cell line: 0>()
----> 1 transform_coarse_dropout = aug.RandomContrast(limit = 0.5,p = 1)
      2 plot_augmentation("999329392.jpg", transform_coarse_dropout)

AttributeError: module 'albumentations' has no attribute 'RandomContrast'

## === cell 29
transform_coarse_dropout = aug.Cutout(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1981192003.py in <cell line: 0>()
----> 1 transform_coarse_dropout = aug.Cutout(p=1)
      2 plot_augmentation("999329392.jpg", transform_coarse_dropout)

AttributeError: module 'albumentations' has no attribute 'Cutout'

## === cell 30
transform_coarse_dropout = aug.Transpose(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)

## === cell 31
img_width, img_height = 224, 224


## === cell 32
train = pd.read_csv(general_path + 'train.csv')
train['label'] = train['label'].astype('string')

## === cell 33
datagen = ImageDataGenerator(validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
                    )

                             
train_datagen_flow = datagen.flow_from_dataframe(dataframe=train,
                                                 directory=general_path + 'train_images',
                                                 x_col='image_id',
                                                 y_col='label',
                                                 target_size=(img_width, img_height),
                                                 batch_size=64,
                                                 subset='training')

## === cell 34
datagen2 = ImageDataGenerator(validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
                   )

valid_datagen_flow = datagen2.flow_from_dataframe( 
    dataframe=train,
    directory=general_path + 'train_images', 
    x_col='image_id',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=64,
    subset='validation')

## === cell 35
x,y = train_datagen_flow.next()
for i in range(0,1):
    image = x[i]/255
    plt.imshow(image)
   
    
    


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2228238556.py in <cell line: 0>()
----> 1 x,y = train_datagen_flow.next()
      2 for i in range(0,1):
      3     image = x[i]/255
      4     plt.imshow(image)
      5 

AttributeError: 'DataFrameIterator' object has no attribute 'next'

## === cell 36
from tensorflow.keras.models import load_model
model=load_model('../input/the-model/my_model.h5')

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2063517939.py in <cell line: 0>()
      1 from tensorflow.keras.models import load_model
----> 2 model=load_model('../input/the-model/my_model.h5')

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/the-model/my_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 37
import keras.preprocessing.image

## === cell 38
preds = []
ss = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

for image in ss.image_id:
    img = keras.preprocessing.image.load_img('../input/cassava-leaf-disease-classification/test_images/' + image)
    img = keras.preprocessing.image.img_to_array(img)
    img = keras.preprocessing.image.smart_resize(img, (img_width, img_height))
    img = np.expand_dims(img, 0)
    prediction = model.predict(img)
    preds.append(np.argmax(prediction))

my_submission = pd.DataFrame({'image_id': ss.image_id, 'label': preds})
my_submission.to_csv('submission.csv', index=False) 

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3159011014.py in <cell line: 0>()
      7     img = keras.preprocessing.image.smart_resize(img, (img_width, img_height))
      8     img = np.expand_dims(img, 0)
----> 9     prediction = model.predict(img)
     10     preds.append(np.argmax(prediction))
     11 

NameError: name 'model' is not defined

## === cell 39
my_submission

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3983424916.py in <cell line: 0>()
----> 1 my_submission

NameError: name 'my_submission' is not defined
