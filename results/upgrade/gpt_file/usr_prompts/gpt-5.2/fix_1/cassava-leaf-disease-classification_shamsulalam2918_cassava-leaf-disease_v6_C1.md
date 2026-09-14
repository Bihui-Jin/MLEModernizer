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

0.1000302206104563

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
import glob
import shutil
import cv2
import os
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import Input

%matplotlib inline
plt.rcParams["figure.figsize"] = (17, 6) # (w, h)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2955667233.py in <cell line: 0>()
      3 import cv2
      4 import os
----> 5 from keras_preprocessing import image
      6 from keras_preprocessing.image import ImageDataGenerator
      7 import matplotlib.pyplot as plt

ModuleNotFoundError: No module named 'keras_preprocessing'

## === cell 2
TRAINING_DIR = "../input/cassava-leaf-disease-classification/train_images"
TRAINING_CSV = "../input/cassava-leaf-disease-classification/train.csv"
JSON_LABELS = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"


## === cell 3
train_df = pd.read_csv(TRAINING_CSV)
train_df["label"] = train_df["label"].astype("string") # for Keras flow_from_dataframe

## === cell 4
total_images_count = len(train_df.index)
total_train_img_count = int(len(train_df.index) * 0.8)
total_val_img_count = total_images_count - total_train_img_count
print("Expected images coutns:")
print("\nTotal Images from original directory: {}".format(total_images_count))
print("Training Images: {}".format(total_train_img_count))
print("Validation Images: {}".format(total_val_img_count))

## === cell 5
label_df = pd.read_json(JSON_LABELS, orient = 'index')
label_df = label_df.values.flatten().tolist()
label_df

## === cell 6
train_label_0 = train_df[train_df["label"]== "0"]
train_label_1 = train_df[train_df["label"]== "1"]
train_label_2 = train_df[train_df["label"]== "2"]
train_label_3 = train_df[train_df["label"]== "3"]
train_label_4 = train_df[train_df["label"]== "4"]
len(train_label_4)

## === cell 7
training_images_dir = TRAINING_DIR + "/*.jpg"
print(training_images_dir)
training_images = glob.glob(training_images_dir)
plt.figure(figsize=(12, 12))    
for i in range(1, 10):
    training_image = np.random.choice(training_images)
    training_image_RGB = cv2.imread(training_image)[...,::-1]
    print(training_image_RGB.shape)
    plt.subplot(3, 3, i)
    plt.imshow(training_image_RGB)
    plt.axis('off')

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2125009375.py in <cell line: 0>()
      2 print(training_images_dir)
      3 training_images = glob.glob(training_images_dir)
----> 4 plt.figure(figsize=(12, 12))
      5 for i in range(1, 10):
      6     training_image = np.random.choice(training_images)

NameError: name 'plt' is not defined

## === cell 8
training_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 100,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.3,
    brightness_range = [0.7, 1.4],
    horizontal_flip = True,
    vertical_flip=True,
    fill_mode = "nearest",
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale = 1/255,
    validation_split=0.2
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2374028350.py in <cell line: 0>()
----> 1 training_datagen = ImageDataGenerator(
      2     rescale = 1/255,
      3     rotation_range = 100,
      4     width_shift_range = 0.2,
      5     height_shift_range = 0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
BATCH_SIZE = 24
IMG_WIDTH = 300
IMG_HEIGHT = 300
CHANNEL = 3

print("\nTraining Dataset")
train_ds = training_datagen.flow_from_dataframe(
    train_df,
    TRAINING_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    x_col = "image_id",
    y_col = "label",
    shuffle = True,
    subset = "training"

)
print("\nValidation Dataset")
validation_ds = validation_datagen.flow_from_dataframe(
    train_df,
    TRAINING_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    x_col = "image_id",
    y_col = "label",
    shuffle = False,
    subset = "validation"
)
print("\nClass Indices:")
print(train_ds.class_indices)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3698738350.py in <cell line: 0>()
      5 
      6 print("\nTraining Dataset")
----> 7 train_ds = training_datagen.flow_from_dataframe(
      8     train_df,
      9     TRAINING_DIR,

NameError: name 'training_datagen' is not defined

## === cell 10
class theCallBacks(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if((logs.get("val_accuracy")>0.92) and (logs.get("accuracy")>0.92)): 
            print("\Training Accuracy> 0.92 & Validation Accuracy> 0.92\nCancelling training!")
            self.model.stop_training = True

            
callback_on_metrics = theCallBacks() #Instantiate theCallBacks

reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5,
                    patience= 2, 
                    verbose = 1,
                    cooldown = 1,
                    min_lr=0.0001)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1451369589.py in <cell line: 0>()
----> 1 class theCallBacks(Callback):
      2     def on_epoch_end(self, epoch, logs={}):
      3         if((logs.get("val_accuracy")>0.92) and (logs.get("accuracy")>0.92)):
      4             print("\Training Accuracy> 0.92 & Validation Accuracy> 0.92\nCancelling training!")
      5             self.model.stop_training = True

NameError: name 'Callback' is not defined

## === cell 11
loss_func = CategoricalCrossentropy()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154269738.py in <cell line: 0>()
----> 1 loss_func = CategoricalCrossentropy()
      2 #optimizer = Adam(learning_rate=0.001)

NameError: name 'CategoricalCrossentropy' is not defined

## === cell 12
import datetime
class LearningRateLogger(Callback):
     def __init__(self):
         super().__init__()
         self._supports_tf_logs = True

     def on_epoch_end(self, epoch, logs=None):
         if logs is None or "learning_rate" in logs:
             return
         logs["learning_rate"] = self.model.optimizer.lr
        
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)     


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2336344303.py in <cell line: 0>()
      1 import datetime
----> 2 class LearningRateLogger(Callback):
      3      def __init__(self):
      4          super().__init__()
      5          self._supports_tf_logs = True

NameError: name 'Callback' is not defined

## === cell 13
new_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL))

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/346352065.py in <cell line: 0>()
----> 1 new_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL))

NameError: name 'Input' is not defined

## === cell 14
!pip install git+https://github.com/rcmalli/keras-vggface.git

## === cell 15
!pip install keras_applications

## === cell 16
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, Dropout, MaxPooling2D
from keras.engine import Model

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
DROPOUT_RATE = 0.5
FROZEN_LAYER_NUM = 201
base_model = VGGFace(model='senet50', include_top=False,input_tensor=new_input, pooling='avg')
last_layer = base_model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(4096, activation='relu', name='fc6')(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(1024, activation='relu', name='fc7')(x)
x = Dropout(DROPOUT_RATE)(x)

batch_norm_indices = [2, 6, 9, 12, 21, 25, 28, 31, 42, 45, 48, 59, 62, 65, 74, 78, 81, 84, 95, 98, 101, 112, 115, 118, 129, 132, 135, 144, 148, 151, 154, 165, 168, 171, 182, 185, 188, 199, 202, 205, 216, 219, 222, 233, 236, 239, 248, 252, 255, 258, 269, 272, 275]
for i in range(FROZEN_LAYER_NUM):
    if i not in batch_norm_indices:
        base_model.layers[i].trainable = False

out = Dense(5, activation='softmax', name='classifier')(x)

model = Model(base_model.input, out)

SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001
sgd = keras.optimizers.SGD(lr=SGD_LEARNING_RATE, momentum=0.9, decay=SGD_DECAY, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1020213743.py in <cell line: 0>()
      1 DROPOUT_RATE = 0.5
      2 FROZEN_LAYER_NUM = 201
----> 3 base_model = VGGFace(model='senet50', include_top=False,input_tensor=new_input, pooling='avg')
      4 last_layer = base_model.get_layer('avg_pool').output
      5 x = Flatten(name='flatten')(last_layer)

NameError: name 'VGGFace' is not defined

## === cell 18
num_epochs = 10
steps_per_epoch = total_train_img_count // BATCH_SIZE

## === cell 19
model_checkpoint_path="../input/cassava-model"

checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/148584233.py in <cell line: 0>()
      1 model_checkpoint_path="../input/cassava-model"
      2 
----> 3 checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=../input/cassava-model

## === cell 20
import tensorflow as tf

model = tf.keras.models.load_model("../input/cassava-model/cassava_Model_Senet.h5")
predicted = []
sample_submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

for image in sample_submission.image_id:
    img = tf.keras.preprocessing.image.load_img('../input/cassava-leaf-disease-classification/test_images/2216849948.jpg')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = tf.reshape(img, (-1, IMG_WIDTH, IMG_HEIGHT, CHANNEL))
    prediction = model.predict(img/255)  
    predicted.append(np.argmax(prediction))

submission = pd.DataFrame({'image_id': sample_submission.image_id, 'label': predicted})
submission.to_csv('submission.csv', index=False) 

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3805907218.py in <cell line: 0>()
      1 import tensorflow as tf
      2 
----> 3 model = tf.keras.models.load_model("../input/cassava-model/cassava_Model_Senet.h5")
      4 predicted = []
      5 sample_submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/cassava-model/cassava_Model_Senet.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
