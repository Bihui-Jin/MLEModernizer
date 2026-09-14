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

0.8768510123904503

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.callbacks import ModelCheckpoint

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
training_mode = False
previously_trained_model_file =  '../input/best-model/best_model.h5'
answer_to_life = 42
first_time_train = False

## === cell 3
project_folder = '../input/cassava-leaf-disease-classification'
df = pd.read_csv(f'{project_folder}/train.csv')
df['image_path'] = df['image_id'].apply(lambda x : f'{project_folder}/train_images/{x}')

## === cell 4
df.head()

## === cell 5
from PIL import Image
image1 = Image.open(df['image_path'].tolist()[0])

## === cell 6
import numpy as np
def get_random_crops(img):
    imgs = []
    for i in range(5):
        cropped_img = tf.image.random_crop(img, size = [512,512,3])
        imgs.append(cropped_img)
    return imgs
        
images = get_random_crops(np.array(image1))

## === cell 8
def get_augmented_images(img):
    img = np.array(img)
    imgs = []
    cropped_images = get_random_crops(img)
    for cropped_img in cropped_images:
        

        horizontal_flip = np.random.rand() > 0.5
        vertical_flip   = np.random.rand() > 0.5
        
        aug_img = tf.keras.preprocessing.image.random_shear(cropped_img, 0.20)
        if horizontal_flip: aug_img = tf.image.flip_left_right(aug_img)
        if vertical_flip: aug_img = tf.image.flip_up_down(aug_img)
            
            
        imgs.append(aug_img)
    return imgs
        

## === cell 9
images = get_augmented_images(image1)
tmp_img = image1.resize((512,512))
tmp_img = np.array(tmp_img)
images.append(tmp_img)

## === cell 10
np.array(images).shape

## === cell 11
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1,6, figsize=(20,20)) 
for i in range(6):
    axarr[i].imshow(images[i])


## === cell 13
datagen = ImageDataGenerator(rescale = 1.0/255.0, horizontal_flip = True,shear_range = 0.2, rotation_range=25,
                                channel_shift_range =0.2,zoom_range=0.2, height_shift_range =0.2, 
                                     vertical_flip = True, validation_split = 0.2)

## === cell 14
df['label'] = df['label'].astype(str)
img_size = 512

train_datagen = datagen.flow_from_dataframe(df, x_col = 'image_path', y_col = 'label', batch_size = 8, 
                                               class_mode = 'categorical', target_size = (img_size, img_size),
                                                  seed = answer_to_life, subset = 'training')

## === cell 15
val_datagen = datagen.flow_from_dataframe(df, x_col = 'image_path', y_col = 'label', batch_size = 8, 
                                            class_mode = 'categorical', target_size = (img_size, img_size),
                                              seed = answer_to_life, subset = 'validation')

## === cell 16
if training_mode:
    def create_pretrained_model():

        pretrained_model = tf.keras.applications.EfficientNetB3(weights = '../input/efficientnet-model-file/efficientnetb3_notop.h5', 
                                                                    include_top=False,drop_connect_rate=0.4,  
                                                                        input_shape=(img_size, img_size, 3))
        x = layers.GlobalAveragePooling2D()(pretrained_model.output)

        x = layers.Dense(512, activation= 'relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(5, activation='softmax')(x)


        model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=x)

        decay_steps = int(round(17118.0/8.0))*3
        cosine_decay = CosineDecay(initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3)

        callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

        model.compile(optimizer=tf.keras.optimizers.Adam(cosine_decay), loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    model     = create_pretrained_model()
    callbacks = [ModelCheckpoint(filepath='best_model_final.h5', monitor='val_loss', save_best_only=True)]



## === cell 17


if not first_time_train:
    model = tf.keras.models.load_model(previously_trained_model_file)
    
if training_mode:
    history   = model.fit(train_datagen, epochs = 6, validation_data = val_datagen, callbacks = callbacks)  



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4227690127.py in <cell line: 0>()
      1 if not first_time_train:
----> 2     model = tf.keras.models.load_model(previously_trained_model_file)
      3 
      4 if training_mode:
      5     #model = tf.keras.models.load_model(previously_trained_model_file)

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/best-model/best_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 20
from PIL import Image

## === cell 21
test_image_path = df['image_path'].tolist()[4]
img = Image.open(test_image_path)
img = img.resize((512,512))

## === cell 22
import numpy as np
img = np.array(img)

## === cell 23
img = img / 255.0

## === cell 24
imgs = get_random_crops(img)
imgs = np.array(imgs)

## === cell 25
preds = model.predict(imgs)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4123917733.py in <cell line: 0>()
----> 1 preds = model.predict(imgs)

NameError: name 'model' is not defined

## === cell 26
preds = np.argmax(np.sum(preds, axis=0))

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1234338288.py in <cell line: 0>()
----> 1 preds = np.argmax(np.sum(preds, axis=0))

NameError: name 'preds' is not defined

## === cell 27
preds

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/222146027.py in <cell line: 0>()
----> 1 preds

NameError: name 'preds' is not defined

## === cell 29
df.head()

## === cell 30
test_folder = '../input/cassava-leaf-disease-classification/test_images'

## === cell 31
import os
test_files = os.listdir(test_folder)

## === cell 32
test_files

## === cell 33
submission_df = pd.DataFrame()
image_names = []
predictions = []

for img_name in test_files:

    tmp_img = Image.open(f'{test_folder}/{img_name}')
    
    aug_imgs = get_augmented_images(tmp_img)
    
    tmp_img = tmp_img.resize((512,512))
    tmp_img = np.array(tmp_img)
    
    aug_imgs.append(tmp_img)
    imgs    = np.array(aug_imgs)
    
    imgs    = imgs/ 255.0
    
    
    
    preds   = model.predict(imgs)
    prediction = np.argmax(np.sum(preds, axis=0))
    image_names.append(img_name)
    predictions.append(prediction)
    
submission_df = pd.DataFrame( data = {'image_id' : image_names, 'label' : predictions})
    

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1399172099.py in <cell line: 0>()
     19 
     20 
---> 21     preds   = model.predict(imgs)
     22     prediction = np.argmax(np.sum(preds, axis=0))
     23     image_names.append(img_name)

NameError: name 'model' is not defined

## === cell 35
submission_df

## === cell 37
submission_df.to_csv("submission.csv", index=False) 

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
