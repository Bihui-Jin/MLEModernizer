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

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import cv2
from PIL import Image
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout, Activation, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7, preprocess_input
from tensorflow.keras.metrics import sparse_categorical_accuracy, sparse_categorical_crossentropy
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from functools import partial

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
path = '/kaggle/input/cassava-leaf-disease-classification'

train_images = os.listdir(os.path.join(path, "train_images"))
print("Total images for Train: ", len(train_images))

with open ('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json') as file:
    classes = json.loads(file.read())
    
print(json.dumps(classes,indent=4))

train_df = pd.read_csv(os.path.join(path, "train.csv"))
train_df.head()

train_df['class'] = train_df['label'].map({int(i) : c for i, c in classes.items()}) 

train_df.head()

## === cell 6

plt.subplots(figsize=(12,8))
ax  = sns.countplot(x='class', data=train_df)

for a in ax.patches:
        ax.annotate('{:1}'.format(a.get_height()),
                    (a.get_x()+0.3, a.get_height()))
plt.xticks(rotation=90)
ax.set_title("classes", fontdict={'fontsize':15})
plt.show();

## === cell 8
def plot_images(class_id, label, images_number,verbose=0):

    plot_list = train_df[train_df["label"] == class_id].sample(images_number)['image_id'].tolist()
    
    if verbose:
        print(plot_list)
        
    labels = [label for i in range(len(plot_list))]
    size = np.sqrt(images_number)
    if int(size)*int(size) < images_number:
        size = int(size) + 1
        
    plt.figure(figsize=(20, 20))
    
    for ind, (image_id, label) in enumerate(zip(plot_list, labels)):
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(os.path.join(path, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")
    
    plt.show()


plot_images(class_id=4, 
    label='Healthy',
    images_number=6,
    verbose=1)

plot_images(class_id=3, 
    label='Cassava Mosaic Disease (CMD)',
    images_number=6,
    verbose=1)

plot_images(class_id=2, 
    label='Cassava Green Mottle (CGM)',
    images_number=6,
    verbose=1)

plot_images(class_id=1, 
    label='Cassava Brown Streak Disease (CBSD)',
    images_number=6,
    verbose=1)

plot_images(class_id=0, 
    label='Cassava Bacterial Blight (CBB)',
    images_number=6,
    verbose=1)

## === cell 12
TARGET_SIZE = (380,380)
BATCH_SIZE = 16
STEPS_PER_EPOCH = len(train_df)*0.8 // BATCH_SIZE
VALIDATION_STEPS = len(train_df)*0.2 // BATCH_SIZE
EPOCHS = 5 # would be 10 -> but there is no more free GPU left, I will rerun it next week.

## === cell 14
train_df.label = train_df.label.astype(str)


train_datagen = ImageDataGenerator(validation_split = 0.2,
                                    rotation_range = 45, 
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    fill_mode = 'nearest',
                                    height_shift_range = 0.2,
                                    width_shift_range = 0.2,
                                  )

train_generator = train_datagen.flow_from_dataframe(train_df,
                         directory = '/kaggle/input/cassava-leaf-disease-classification/train_images/',
                         subset = "training",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = TARGET_SIZE,
                         batch_size = BATCH_SIZE,
                         class_mode = "sparse",
                         seed = 2021,
                         shuffle= True)


validation_datagen = ImageDataGenerator(validation_split = 0.2) # no data augmentation on validation set

validation_generator = validation_datagen.flow_from_dataframe(train_df,
                         directory = '/kaggle/input/cassava-leaf-disease-classification/train_images/',
                         subset = "validation",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = TARGET_SIZE,
                         batch_size = BATCH_SIZE,
                         class_mode = "sparse",
                         seed = 2021,
                         shuffle= True)

## === cell 17

weights_path = '/kaggle/input/efficientnetb0/efficientnetb0_notop.h5'

basemodel = EfficientNetB0(
    weights=weights_path, 
    include_top=False,
    input_shape=TARGET_SIZE+(3,))

headmodel = layers.GlobalAveragePooling2D()(basemodel.output)
headmodel = layers.Dense(5, activation="softmax")(headmodel) # 5 -> for five classes
model = keras.Model(inputs=basemodel.input, outputs=headmodel)

model.load_weights("../input/best-weights-efficient/best.h5")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/357228586.py in <cell line: 0>()
      3 weights_path = '/kaggle/input/efficientnetb0/efficientnetb0_notop.h5'
      4 
----> 5 basemodel = EfficientNetB0(
      6     weights=weights_path,
      7     include_top=False,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    569     name="efficientnetb0",
    570 ):
--> 571     return EfficientNet(
    572         1.0,
    573         1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)
    273 
    274     if not (weights in {"imagenet", None} or file_utils.exists(weights)):
--> 275         raise ValueError(
    276             "The `weights` argument should be either "
    277             "`None` (random initialization), `imagenet` "

ValueError: The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.

## === cell 18
"""
weights_path = '/kaggle/input/efficientnetb0/efficientnetb0_notop.h5'

basemodel = EfficientNetB0(
    weights=weights_path, 
    include_top=False,
    input_shape=TARGET_SIZE+(3,))

headmodel = layers.GlobalAveragePooling2D()(basemodel.output)
headmodel = layers.Dense(5, activation="softmax")(headmodel)
model = keras.Model(inputs=basemodel.input, outputs=headmodel)
"""

## === cell 20
"""

model_save = ModelCheckpoint('./best_weights.h5', 
                             save_best_only = True, 
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 2, min_lr = 1e-6, 
                              mode = 'min', verbose = 1)
early_stop = EarlyStopping(monitor = 'val_loss', 
                           patience = 3, mode = 'min', verbose = 1,
                           restore_best_weights = True)


model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs = EPOCHS, 
    validation_data = validation_generator,
    validation_steps = VALIDATION_STEPS,
    callbacks = [model_save, early_stop, reduce_lr],
)

model.save("model.h5")

"""

## === cell 21
""" 
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
     
    if len(loss_list) == 0: 
        print('Loss is missing in history') 
        return 
     
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list: 
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


plot_history(history)
"""

## === cell 23
ss = pd.read_csv(os.path.join('/kaggle/input/cassava-leaf-disease-classification', "sample_submission.csv"))
preds = []
results = []

for image_id in ss.image_id:
    image = Image.open(os.path.join('/kaggle/input/cassava-leaf-disease-classification', "test_images", image_id))
    image = image.resize(TARGET_SIZE)
    image = np.expand_dims(image, axis = 0)
    preds.append(np.argmax(model.predict(image)))
    res = max(set(preds), key = preds.count)
    results.append(res)

ss['label'] = results
ss.to_csv('submission.csv', index = False)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3992993214.py in <cell line: 0>()
      7     image = image.resize(TARGET_SIZE)
      8     image = np.expand_dims(image, axis = 0)
----> 9     preds.append(np.argmax(model.predict(image)))
     10     res = max(set(preds), key = preds.count)
     11     results.append(res)

NameError: name 'model' is not defined
