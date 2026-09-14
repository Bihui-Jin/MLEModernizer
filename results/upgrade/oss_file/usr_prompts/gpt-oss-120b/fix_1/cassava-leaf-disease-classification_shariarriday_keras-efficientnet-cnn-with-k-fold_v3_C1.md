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

0.8559987911755818

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow import image, cast, float32
from tensorflow.io import decode_image, read_file
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,Input,Dense,GlobalAveragePooling2D,Dropout,BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow import numpy_function
from tensorflow.keras.preprocessing.image import load_img,img_to_array

from sklearn.model_selection import StratifiedKFold

import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os
import glob

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

BATCH_SIZE = 64
ROW = 224
COL = 224

train_csv_loc = '../input/cassava-leaf-disease-classification/train.csv'
train_location = '../input/cassava-leaf-disease-classification/train_images/'
test_location = '../input/cassava-leaf-disease-classification/test_images'

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def augmentations(file):

    file = read_file(file)
    
    file = image.decode_jpeg(file, channels=3)

    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

    file = transform(image=file.numpy())['image']
    
    file = preprocess_input(file)

    file = image.resize(file, [ROW, COL])

    return file

## === cell 4
def fetch_image_without_aug(filename , label):
    
    image_file = read_file(filename)

    image_file = image.decode_jpeg(image_file, channels=3)

    image_file = preprocess_input(image_file)

    image_file = image.resize(image_file, [ROW, COL])

    return image_file, label

## === cell 5
def fetch_image_with_aug(filename , label):

    aug_img = numpy_function(func=augmentations, inp=[filename], Tout=float32)

    return aug_img, label

## === cell 6
def getDatasetFromDataframe(train_files, train_labels, val_files, val_labels):

    train_ds = Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(len(train_files))
    train_ds = train_ds.map(fetch_image_with_aug , num_parallel_calls=16)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.shuffle(len(val_files))
    val_ds = val_ds.map(fetch_image_without_aug , num_parallel_calls=16)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    return train_ds , val_ds

## === cell 8
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)

## === cell 9
def create_callbacks(Folder_name):
    if not os.path.exists(os.path.join("Weights" , Folder_name)):
        os.mkdir(os.path.join("Weights", Folder_name))

    if not os.path.exists(os.path.join("Weights", "logs" , Folder_name)):
        os.mkdir(os.path.join("Weights", "logs", Folder_name))

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save = ModelCheckpoint(os.path.join("Weights", Folder_name),
        monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False)

    weight_save_only = ModelCheckpoint(os.path.join("Weights", Folder_name + ".h5"),
        monitor="val_accuracy", verbose=0, save_best_only=True, save_weights_only=True)

    tensorboard = TensorBoard(os.path.join(
        "Weights", "logs", Folder_name), histogram_freq=1)

    callbacks = [lr_scheduler, weight_save, tensorboard, weight_save_only]

    return callbacks,histories


## === cell 11
def create_model(training = True, weights = "imagenet"):
    base_model = EfficientNetB0(weights=weights , include_top=False , input_shape=(ROW,COL,3))

    x = Input(shape = (ROW,COL,3))
    out_1 = base_model(x,training = training)
    out_1 = GlobalAveragePooling2D(name = 'encoding')(out_1)
    out_1 = Dropout(0.5)(out_1)
    output = Dense(5 , activation="softmax")(out_1)

    final_model = Model(inputs = x , outputs = output)

    final_model.summary()

    return final_model

## === cell 13
train_csv = pd.read_csv(train_csv_loc)
train_csv['image_id'] = train_csv['image_id'].map(lambda x: train_location+x)

Folder_name = "Exp_"

create_k_folds = StratifiedKFold(n_splits=5)

fold_number = 1

for train_index, test_index in create_k_folds.split(train_csv['image_id'] , train_csv['label']):

    print("Fold Number {} is starting it's training".format(fold_number))
    print("Creating Callbacks...")
    Folder_name = Folder_name + str(fold_number)
    callbacks, histories = create_callbacks(Folder_name)
    
    print("Importing Datasets...")
    X_train, X_test = train_csv['image_id'].loc[train_index], train_csv['image_id'].loc[test_index]
    y_train, y_test = train_csv['label'].loc[train_index], train_csv['label'].loc[test_index]

    train_ds, val_ds = getDatasetFromDataframe(X_train, y_train, X_test, y_test)

    final_model = create_model()

    final_model.compile(optimizer=RMSprop(learning_rate=1e-4),
                    loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    try:
        print("Starting training...")
        hist = final_model.fit(train_ds, epochs=15,
                            validation_data=val_ds, callbacks=callbacks, workers=16)
    except:
        print("Error...")

    fold_number += 1

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4242405064.py in <cell line: 0>()
     14     print("Creating Callbacks...")
     15     Folder_name = Folder_name + str(fold_number)
---> 16     callbacks, histories = create_callbacks(Folder_name)
     17 
     18     print("Importing Datasets...")

/tmp/ipykernel_11/3996997898.py in create_callbacks(Folder_name)
      1 def create_callbacks(Folder_name):
      2     if not os.path.exists(os.path.join("Weights" , Folder_name)):
----> 3         os.mkdir(os.path.join("Weights", Folder_name))
      4 
      5     if not os.path.exists(os.path.join("Weights", "logs" , Folder_name)):

FileNotFoundError: [Errno 2] No such file or directory: 'Weights/Exp_1'

## === cell 15
checkpoints = glob.glob(os.path.join('../input/efficient-net-weights' , 'Exp_*.h5'))

Models = []
for checkpoint in checkpoints:
    new_model = create_model(training = False, weights = None)
    new_model.load_weights(checkpoint)
    Models.append(new_model)

print('Models loaded...')

images = os.listdir(test_location)
results = []

for single in images:

    image = load_img(test_location+'/'+single , target_size=(ROW,COL))
    image = img_to_array(image)
    image = preprocess_input(image)
    batch = np.array([image])
    res = []
    for model in Models:
        res.append(model.predict(batch))
    res = np.mean(res , 0)
    results.append({"image_id" : single , "label" : np.argmax(res)})

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2347334329.py in <cell line: 0>()
     14 for single in images:
     15 
---> 16     image = load_img(test_location+'/'+single , target_size=(ROW,COL))
     17     image = img_to_array(image)
     18     image = preprocess_input(image)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/cassava-leaf-disease-classification/test_images/test_images'

## === cell 16
submission = pd.DataFrame(results)
submission.to_csv('submission.csv' , index = False)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
