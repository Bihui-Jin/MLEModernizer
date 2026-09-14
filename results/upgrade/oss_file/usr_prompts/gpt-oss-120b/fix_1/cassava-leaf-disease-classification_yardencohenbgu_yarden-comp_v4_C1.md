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

0.6128

# 6. Current score

0.13191

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd

labels = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
labels.head()


## === cell 1
train_dir = '../input/cassava-leaf-disease-classification/train_images'
test_dir = '../input/cassava-leaf-disease-classification/test_images'

batch_size = 32
img_height = 256
img_width = 256

from keras import preprocessing


train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./225,
                                                       rotation_range=40,
                                                       width_shift_range=0.2,
                                                       height_shift_range=0.2,
                                                       shear_range=0.2,
                                                       zoom_range=0.2,
                                                       validation_split=0.3,
                                                       horizontal_flip=True,)

test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_dataframe(dataframe=labels.astype(str),
                                             directory=train_dir,
                                             subset='training',
                                             x_col="image_id",
                                             validation_split=0.2,
                                             y_col="label",
                                             shuffle=True,
                                             target_size=(img_height,img_width),
                                             batch_size=batch_size,
                                             class_mode='categorical')

val_ds = train_datagen.flow_from_dataframe(dataframe=labels.astype(str),
                                             directory=train_dir,
                                             subset='validation',
                                             x_col="image_id",
                                             validation_split=0.2,
                                             y_col="label",
                                             shuffle=True,
                                             target_size=(img_height,img_width),
                                             batch_size=batch_size,
                                             class_mode='categorical')


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import numpy as np
np.unique(train_ds.labels)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3397218765.py in <cell line: 0>()
      1 import numpy as np
----> 2 np.unique(train_ds.labels)

NameError: name 'train_ds' is not defined

## === cell 5
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2311705460.py in <cell line: 0>()
----> 1 for image_batch, labels_batch in train_ds:
      2   print(image_batch.shape)
      3   print(labels_batch.shape)
      4   break

NameError: name 'train_ds' is not defined

## === cell 7
num_classes = 5

from keras import models, layers, optimizers

model = models.Sequential()

model.add(layers.Conv2D(16,(3,3), activation='relu', input_shape=(img_height,img_width,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(img_height,img_width,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation='relu', input_shape=(img_height,img_width,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=optimizers.Adam(),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


## === cell 9
import tensorflow as tf

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)


## === cell 10
epochs=10
steps_per_epoch = 10

history = model.fit(
  train_ds,steps_per_epoch=steps_per_epoch,
  validation_data=val_ds,validation_steps=steps_per_epoch,
  epochs=epochs
)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3700183343.py in <cell line: 0>()
      3 
      4 history = model.fit(
----> 5   train_ds,steps_per_epoch=steps_per_epoch,
      6   validation_data=val_ds,validation_steps=steps_per_epoch,
      7   epochs=epochs

NameError: name 'train_ds' is not defined

## === cell 11
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2922392303.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
      2 
----> 3 acc = history.history['accuracy']
      4 val_acc = history.history['val_accuracy']
      5 

NameError: name 'history' is not defined

## === cell 12
def generate_number():

def generate_layer_type():

def generate_layer_params(layer_type):

def generate_layer(layer_type,layer_params):

def create_model(layers):

def train_and_eval_model(model):

def log_score(score):

def print_score(score):


## --- ERROR in cell 12, traceback:
  File "/tmp/ipykernel_11/2751098604.py", line 3
    def generate_layer_type():
    ^
IndentationError: expected an indented block after function definition on line 1


## === cell 13
best_score = 0
while True:
    N_layers = generate_number()
    layers = []
    for i in range(N):
        layer_type = generate_layer_type()
        layer_params = generate_layer_params(layer_type)
        layer = generate_layer(layer_type,layer_params)
        layers.append(layer)
        
    model = create_model(layers)
    score = train_and_eval_model(model)
    log_score(score)
    if score>best_score:
        print_score(score)
        best_score = score
    
    


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2143039034.py in <cell line: 0>()
      1 best_score = 0
      2 while True:
----> 3     N_layers = generate_number()
      4     layers = []
      5     for i in range(N):

NameError: name 'generate_number' is not defined

## === cell 14
epochs = 25
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


print(model.summary())

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4166953919.py in <cell line: 0>()
      7 print(model.summary())
      8 
----> 9 numpy.random.seed(seed)
     10 model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
     11 

NameError: name 'numpy' is not defined

## === cell 15
test_dir = '../input/cassava-leaf-disease-classification/test_images'

from keras import preprocessing


test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2268500078.py in <cell line: 0>()
      4 
      5 
----> 6 test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

AttributeError: module 'keras.api.preprocessing.image' has no attribute 'ImageDataGenerator'

## === cell 16
from PIL import Image
import numpy as np
import pandas as pd
import os

ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
preds = []

for image_id in ss.image_id:
    image = Image.open(os.path.join(test_dir, image_id))
    image = image.resize((img_height, img_width))
    image = np.expand_dims(image, axis = 0)
    preds.append(np.argmax(model.predict(image)))

ss['label'] = preds


## === cell 17
ss.to_csv('submission.csv', index = False)
