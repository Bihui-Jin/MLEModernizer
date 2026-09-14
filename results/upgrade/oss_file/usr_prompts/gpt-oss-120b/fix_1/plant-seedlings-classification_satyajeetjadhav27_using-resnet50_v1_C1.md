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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
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

0.91813

# 6. Current score

0.13514

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
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_dir = '/kaggle/input/plant-seedlings-classification/train'
test_dir = "/kaggle/input/plant-seedlings-classification/"
img_size = 224
batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                            rotation_range=30,
                            brightness_range=[0.5,1.2],
                            horizontal_flip=True,
                            validation_split=0.25,
                            zoom_range=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')


train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(img_size,img_size),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              subset='training',
                                              class_mode='categorical')


val_generator = datagen.flow_from_directory(train_dir,
                                            target_size=(img_size,img_size),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            subset='validation',
                                            class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    classes=['test'],
    target_size=(img_size, img_size),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')


## === cell 3
label = [k for k in train_generator.class_indices]
samples = train_generator.__next__()
images = samples[0]
titles = samples[1]
plt.figure(figsize=(20,20))

for i in range(20):
    plt.subplot(5,5,i+1)
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    plt.imshow(images[i])
    plt.title(f"Class: {label[np.argmax(titles[i],axis=0)]}")
    plt.axis("off")


## === cell 4
base_model_resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))


## === cell 5
"""for layer in base_model_resnet50.layers:
    print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")"""
len(base_model_resnet50.layers)


## === cell 6
for layer in base_model_resnet50.layers[:50]:
    layer.trainable = False


## === cell 7
for layer in base_model_resnet50.layers:
    print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")
len(base_model_resnet50.layers)


## === cell 8
model_resnet50 = tf.keras.models.Sequential(layers=[
    base_model_resnet50,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(12, activation='softmax')
])


## === cell 9
model_resnet50.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])


## === cell 10
model_name = "model_resnet50.h5"
Checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lrr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.3, min_lr=0.00000001)
cb_List= [Checkpoint, es, lrr]


## === cell 11
EPOCH = 50
history_resnet50 = model_resnet50.fit(train_generator, epochs=EPOCH, validation_data=val_generator, callbacks=cb_List)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1620597841.py in <cell line: 0>()
      1 # Train the model
      2 EPOCH = 50
----> 3 history_resnet50 = model_resnet50.fit(train_generator, epochs=EPOCH, validation_data=val_generator, callbacks=cb_List)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/nn.py in categorical_crossentropy(target, output, from_logits, axis)
    658     for e1, e2 in zip(target.shape, output.shape):
    659         if e1 is not None and e2 is not None and e1 != e2:
--> 660             raise ValueError(
    661                 "Arguments `target` and `output` must have the same shape. "
    662                 "Received: "

ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 13), output.shape=(None, 12)

## === cell 12
accuracy = history_resnet50.history['accuracy']
val_accuracy = history_resnet50.history['val_accuracy']
loss = history_resnet50.history['loss']
val_loss = history_resnet50.history['val_loss']

num_epochs = len(accuracy)
epochs = list(range(1, num_epochs+1))

plt.figure(figsize=(20, 8))

plt.plot(epochs, accuracy, label='Training Accuracy', color='blue')

plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='green')

plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/752272459.py in <cell line: 0>()
      1 # Extract accuracy and loss from history
----> 2 accuracy = history_resnet50.history['accuracy']
      3 val_accuracy = history_resnet50.history['val_accuracy']
      4 loss = history_resnet50.history['loss']
      5 val_loss = history_resnet50.history['val_loss']

NameError: name 'history_resnet50' is not defined

## === cell 13
plt.figure(figsize=(20, 8))
plt.plot(epochs, loss, label='Training Loss', color='red')

plt.plot(epochs, val_loss, label='Validation Loss', color='purple')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3807502519.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 8))
      2 # Plot loss
----> 3 plt.plot(epochs, loss, label='Training Loss', color='red')
      4 
      5 # Plot validation loss

NameError: name 'epochs' is not defined

## === cell 14
y_test = val_generator.classes
y_pred = model_resnet50.predict(val_generator, batch_size= 32)
y_pred = np.argmax(y_pred,axis=1)


## === cell 15
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred,target_names = label))


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4080954297.py in <cell line: 0>()
      1 from sklearn.metrics import classification_report, confusion_matrix
----> 2 print(classification_report(y_test,y_pred,target_names = label))

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in classification_report(y_true, y_pred, labels, target_names, sample_weight, digits, output_dict, zero_division)
   2330             )
   2331         else:
-> 2332             raise ValueError(
   2333                 "Number of classes, {0}, does not match size of "
   2334                 "target_names, {1}. Try specifying the labels "

ValueError: Number of classes, 12, does not match size of target_names, 13. Try specifying the labels parameter

## === cell 16
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

class_labels = label

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


## === cell 17
species_list = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
                "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                "Sugar beet"]
preds = model_resnet50.predict(test_generator, steps=test_generator.samples)
class_list = []
for i in range(preds.shape[0]):
    y_class = preds[i,:].argmax(axis=-1)
    class_list.append(species_list[y_class])
    
submission= pd.DataFrame()
submission['file'] = test_generator.filenames
submission['file'] = submission['file'].str.replace(r'test/', '')
submission['species'] = class_list


## === cell 18
submission.head(5)


## === cell 19
submission.to_csv('submission.csv', index=False)
