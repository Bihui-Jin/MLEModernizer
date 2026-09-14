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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.87096

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = '../train.csv'
train_data_dir = '../input/paddy-disease-classification/train_images'
epochs = 100
lr = 1e-3
valid_split = 0.2
input_size = 224
batch_size = 32
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy


## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(patience=10,
                                              monitor='val_loss',
                                              restore_best_weights=True,
                                              verbose=1)


## === cell 3
def inception(x, filters, projection, init=initializer, name=None):
    f_1x1, f_3x3, f_3x3_reduce, f_5x5, f_5x5_reduce = filters
    x1 = Conv2D(filters=f_1x1,
                kernel_size=(1, 1),
                kernel_initializer=init,
                strides=(1, 1),
                activation=relu,
                padding='same')(x)
    x3_reducer = Conv2D(filters=f_3x3_reduce,
                        kernel_size=(1, 1),
                        kernel_initializer=init,
                        strides=(1, 1),
                        activation=relu,
                        padding='same')(x)
    x5_reducer = Conv2D(filters=f_5x5_reduce,
                        kernel_size=(1, 1),
                        kernel_initializer=init,
                        strides=(1, 1),
                        activation=relu,
                        padding='same')(x)
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)

    x3 = Conv2D(filters=f_3x3,
                kernel_size=(3, 3),
                kernel_initializer=init,
                strides=(1, 1),
                activation=relu,
                padding='same')(x3_reducer)
    x5 = Conv2D(filters=f_5x5,
                kernel_size=(5, 5),
                kernel_initializer=init,
                strides=(1, 1),
                activation=relu,
                padding='same')(x5_reducer)
    proj = Conv2D(filters=projection,
                  kernel_size=(1, 1),
                  kernel_initializer=init,
                  strides=(1, 1),
                  activation=relu,
                  padding='same')(pool)

    x = concatenate([x1, x3, x5, proj], axis=3, name=name)

    return x


def model_builder(shape, classes):
    input_layer = Input(shape=shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), kernel_initializer=initializer, strides=(2, 2), activation=relu,
               padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [64, 128, 96, 32, 16], projection=32, name='inception_3a')
    x = inception(x, [128, 192, 128, 96, 32], projection=64, name='inception_3b')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [192, 208, 96, 48, 16], projection=64, name='inception_4a')

    aux_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
    aux_1 = Conv2D(filters=128, kernel_size=(1, 1), kernel_initializer=initializer, strides=(1, 1), activation=relu,
                   padding='valid')(aux_1)
    aux_1 = Dense(units=1024, activation=relu)(aux_1)
    aux_1 = Dropout(rate=0.7)(aux_1)
    aux_1 = GlobalAveragePooling2D()(aux_1)
    aux_out1 = Dense(units=classes, activation=softmax, name='aux_out1')(aux_1)

    x = inception(x, [160, 224, 112, 64, 24], projection=64, name='inception_4b')
    x = inception(x, [128, 256, 128, 64, 24], projection=64, name='inception_4c')
    x = inception(x, [112, 288, 144, 64, 32], projection=64, name='inception_4d')
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name='inception_4e')

    aux_2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
    aux_2 = Conv2D(filters=128, kernel_size=(1, 1), kernel_initializer=initializer, strides=(1, 1), activation=relu,
                   padding='valid')(aux_2)
    aux_2 = Dense(units=1024, activation=relu)(aux_2)
    aux_2 = Dropout(rate=0.7)(aux_2)
    aux_2 = GlobalAveragePooling2D()(aux_2)
    aux_out2 = Dense(units=classes, activation=softmax, name='aux_out2')(aux_2)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name='inception_5a')
    x = inception(x, [384, 384, 192, 128, 48], projection=128, name='inception_5b')
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = Dropout(rate=0.4)(x)
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(units=classes, activation=softmax, name='main_out')(x)

    model = Model(input_layer, [output_layer, aux_out1, aux_out2])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights={'main_out': 1, 'aux_out1': 0.3, 'aux_out2': 0.3},
                  metrics=['accuracy'])

    return model


## === cell 4
generator = ImageDataGenerator(rescale=1 / 255,
                               validation_split=valid_split)

train_data = generator.flow_from_directory(directory=train_data_dir,
                                           target_size=(input_size, input_size),
                                           batch_size=batch_size,
                                           subset='training')

valid_data = generator.flow_from_directory(directory=train_data_dir,
                                           target_size=(input_size, input_size),
                                           subset='validation')


## === cell 5
len(train_data.next()[0]), len(valid_data.next()[0])


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1437816410.py in <cell line: 0>()
----> 1 len(train_data.next()[0]), len(valid_data.next()[0])

AttributeError: 'DirectoryIterator' object has no attribute 'next'

## === cell 6
model = model_builder(shape=(input_size, input_size, 3), classes=classes)


## === cell 7
model.summary()


## === cell 8
tf.keras.utils.plot_model(model, 'baseline_inception.png')


## === cell 9
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    callbacks=[early_stop])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/327053946.py in <cell line: 0>()
----> 1 history = model.fit(train_data,
      2                     validation_data=valid_data,
      3                     epochs=epochs,
      4                     callbacks=[early_stop])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in build(self, y_true, y_pred)
    606                 flat_loss_weights = tree.flatten(loss_weights)
    607                 if len(tree.flatten(loss)) != len(flat_loss_weights):
--> 608                     raise ValueError(
    609                         f"`loss_weights` must match the number of losses, "
    610                         f"got {len(tree.flatten(loss))} losses "

ValueError: `loss_weights` must match the number of losses, got 1 losses and 3 weights.

## === cell 10
plt.figure(figsize=[12,6], dpi=300)
sns.lineplot(x=list(range(len(history.history['main_out_accuracy']))),
             y=history.history['main_out_accuracy'],
             label='train')
sns.lineplot(x=list(range(len(history.history['val_main_out_accuracy']))),
             y=history.history['val_main_out_accuracy'],
             label='validation')
plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4240111747.py in <cell line: 0>()
      1 plt.figure(figsize=[12,6], dpi=300)
----> 2 sns.lineplot(x=list(range(len(history.history['main_out_accuracy']))),
      3              y=history.history['main_out_accuracy'],
      4              label='train')
      5 sns.lineplot(x=list(range(len(history.history['val_main_out_accuracy']))),

NameError: name 'history' is not defined

## === cell 11
plt.figure(figsize=[12,6], dpi=300)
sns.lineplot(x=list(range(len(history.history['main_out_loss']))),
             y=history.history['main_out_loss'],
             label='train')
sns.lineplot(x=list(range(len(history.history['val_main_out_loss']))),
             y=history.history['val_main_out_loss'],
             label='validation')
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2945853903.py in <cell line: 0>()
      1 plt.figure(figsize=[12,6], dpi=300)
----> 2 sns.lineplot(x=list(range(len(history.history['main_out_loss']))),
      3              y=history.history['main_out_loss'],
      4              label='train')
      5 sns.lineplot(x=list(range(len(history.history['val_main_out_loss']))),

NameError: name 'history' is not defined

## === cell 12
print(f'train score : {model.evaluate(train_data)} -- validation : {model.evaluate(valid_data)}')


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1168347671.py in <cell line: 0>()
----> 1 print(f'train score : {model.evaluate(train_data)} -- validation : {model.evaluate(valid_data)}')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in build(self, y_true, y_pred)
    606                 flat_loss_weights = tree.flatten(loss_weights)
    607                 if len(tree.flatten(loss)) != len(flat_loss_weights):
--> 608                     raise ValueError(
    609                         f"`loss_weights` must match the number of losses, "
    610                         f"got {len(tree.flatten(loss))} losses "

ValueError: `loss_weights` must match the number of losses, got 1 losses and 3 weights.

## === cell 13
temp = pd.DataFrame(history.history)
temp.to_csv('history.csv', index=False)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2394311269.py in <cell line: 0>()
----> 1 temp = pd.DataFrame(history.history)
      2 temp.to_csv('history.csv', index=False)

NameError: name 'history' is not defined

## === cell 14
model.save('baseline.hdf5')


## === cell 15
model.save_weights('baseline_inception_weights.hdf5')


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4010335281.py in <cell line: 0>()
----> 1 model.save_weights('baseline_inception_weights.hdf5')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in save_weights(model, filepath, overwrite, **kwargs)
    222 def save_weights(model, filepath, overwrite=True, **kwargs):
    223     if not str(filepath).endswith(".weights.h5"):
--> 224         raise ValueError(
    225             "The filename must end in `.weights.h5`. "
    226             f"Received: filepath={filepath}"

ValueError: The filename must end in `.weights.h5`. Received: filepath=baseline_inception_weights.hdf5

## === cell 16
train_data.class_indices


## === cell 17
test_preds = []

for i, file in enumerate(os.listdir('../input/paddy-disease-classification/test_images')):
    img = load_img(os.path.join('../input/paddy-disease-classification/test_images',file), target_size=(input_size,input_size))
    pred = model.predict(np.expand_dims(img_to_array(img)/255.0, axis=0))
    pred = np.argmax(pred[0][0])
    label = list(train_data.class_indices.keys())[list(train_data.class_indices.values()).index(pred)]
    test_preds.append([file, label])
    print(f"{i+1}/{len(os.listdir('../input/paddy-disease-classification/test_images'))}", end='\r')


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2359321360.py in <cell line: 0>()
      2 
      3 for i, file in enumerate(os.listdir('../input/paddy-disease-classification/test_images')):
----> 4     img = load_img(os.path.join('../input/paddy-disease-classification/test_images',file), target_size=(input_size,input_size))
      5     pred = model.predict(np.expand_dims(img_to_array(img)/255.0, axis=0))
      6     pred = np.argmax(pred[0][0])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/paddy-disease-classification/test_images/test_images'

## === cell 18
temp = pd.DataFrame(test_preds, columns=['image_id', 'label'])
temp.to_csv('baseline_submission.csv', index=False)
temp


## --- ERROR in outputing the csv:
Invalid submission: Expected 2602 rows, but got 576
