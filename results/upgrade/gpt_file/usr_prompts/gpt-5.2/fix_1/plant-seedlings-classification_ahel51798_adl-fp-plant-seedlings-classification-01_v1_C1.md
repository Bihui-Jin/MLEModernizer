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

3.12

# 3. Installed packages

No external packages required in the script and installed.

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

0.9836272040302269

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
!ls -l /kaggle/input/plant-seedlings-classification




## === cell 1
%%time

from datetime import datetime, timedelta
import gc
import os

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from IPython.display import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.applications import ResNet50

start_time = datetime.now()
print('Time now is', start_time)
end_training_by_tdelta = timedelta(seconds=8400)
this_run_file_prefix = start_time.strftime('%Y%m%d_%H%M_')
print('this_run_file_prefix',this_run_file_prefix)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
%%time

print('TensorFlow version:', tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

## === cell 3
%%time

def define_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.0, # change to use validation instead of training on entire training set
    )

    train_generator = train_datagen.flow_from_directory(
        directory='/kaggle/input/plant-seedlings-classification/train',
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode="categorical",
        subset='training',
    )

    validation_generator = train_datagen.flow_from_directory(
        directory='/kaggle/input/plant-seedlings-classification/train',
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode="categorical",
        subset='validation',
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        directory='/kaggle/input/plant-seedlings-classification/',
        classes=['test'],
        target_size=(width, height),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator


## === cell 4
%%time

def create_ResNet50_model():
    pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,  # Add the pre-trained ResNet50 as the base of the model
        tf.keras.layers.GlobalAveragePooling2D(),  # Add global average pooling layer to reduce feature map size
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')  # Add a Dense layer with softmax activation for classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
      
    

def create_ResNet101V2_model():
    pretrained_model = tf.keras.applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,  # Add the pre-trained ResNet101V2 as the base of the model
        tf.keras.layers.GlobalAveragePooling2D(),  # Add global average pooling layer to reduce feature map size
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')  # Add a Dense layer with softmax activation for classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_VGG16_model():
    pretrained_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True # False

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_Xception_model():
    pretrained_model = tf.keras.applications.Xception(include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_DenseNet_model():
    pretrained_model = tf.keras.applications.DenseNet201(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_EfficientNet_model():
    pretrained_model = efficientnet.EfficientNetB7(weights = 'noisy-student', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_InceptionV3_model():
    pretrained_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_ResNet152_model():
    pretrained_model = tf.keras.applications.ResNet152V2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_MobileNetV2_model():
    pretrained_model = tf.keras.applications.MobileNetV2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_InceptionResNetV2_model():
    pretrained_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = [*IMAGE_SIZE, 3])
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:  
        lr = LR_START + (epoch * (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS)
    elif epoch < (LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS):
        lr = LR_MAX
    else:    
        lr = LR_MIN + (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)
    return lr

def write_history(j):
    history_dict = [0] * no_of_models
    for i in range(j + 1):
        if (historys[i] != 0):
            history_dict[i] = historys[i].history
    filename = './' + this_run_file_prefix + 'model_history_' + str(j) + '.pkl'
    pklfile = open(filename, 'ab')
    pickle.dump(history_dict, pklfile)
    pklfile.close()
    
def load_history(filename):
    with open(filename, 'rb') as file:
        history_dict = pickle.load(file)
    return history_dict

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


## === cell 5
%%time

IMAGE_SIZE = [299, 299]  # ResNet50 default image size
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
width        = 299
height       = 299
num_classes = 12
CLASSES = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
                "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                "Sugar beet"]


train_generator, validation_generator, test_generator = define_generators()

## === cell 6
%%time

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = LR_START
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.80

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(30)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
plt.title("Learning Rate over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.show()

print("lrfn y:", y)

## === cell 7
%%time

no_of_models = 1   
models = [0] * no_of_models  # Initialize a list to store models
start_model = 0
end_model = 1  # Define the range of models to train
model_indx_0 = start_model
model_indx_1 = start_model + 1

EPOCHS = 3
historys = [0] * no_of_models  # Initialize a list to store training history for each model
finished_models = 0

checkpoint_path = '/kaggle/working/best_model'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

val_probabilities = [0] * no_of_models
test_probabilities = [0] * no_of_models
all_probabilities = [0] * no_of_models

with strategy.scope():
    
    for j in range(no_of_models):
        models[j] = create_InceptionV3_model()
    

models[0].summary()

for j in range(start_model, end_model):
    start_training = datetime.now()  # Record the start time of training
    print(start_training)
    time_from_start_program_tdelta = start_training - start_time
    if time_from_start_program_tdelta > end_training_by_tdelta:
        print(j, 'time limit for doing training over, get out')
        break
    print('LR_EXP_DECAY:', LR_EXP_DECAY, '. LR_MAX:', LR_MAX)
    historys[j] = models[j].fit(train_generator,
                                epochs=EPOCHS,
                                steps_per_epoch=train_generator.samples // BATCH_SIZE,
                                validation_data= validation_generator, 
                                validation_steps=validation_generator.samples // BATCH_SIZE,
                                callbacks=[learning_rate_reduction, early_stopping, lr_callback, model_checkpoint_callback])
    
    write_history(j)
    filename = this_run_file_prefix + 'models_' + str(j)
    models[j].save(filename)
    gc.collect()
    finished_models = j + 1  # Update the count of finished models

print(datetime.now())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/kaggle/working/best_model

## === cell 8
%%time

history_to_analyze = historys[j].history

plot_history(history_to_analyze)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'j' is not defined

## === cell 9
filename = '/kaggle/input/fork-of-plant-seedlings-classification-01-v3-model/20231203_1610_model_history_0.pkl'
loaded_history = load_history(filename)

history_to_analyze = loaded_history[0]

plot_history(history_to_analyze)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/210325698.py in <cell line: 0>()
      1 # Load the history from the saved pickle file
      2 filename = '/kaggle/input/fork-of-plant-seedlings-classification-01-v3-model/20231203_1610_model_history_0.pkl'
----> 3 loaded_history = load_history(filename)
      4 
      5 # Assume the history for the first model is what you want to analyze

<timed exec> in load_history(filename)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/fork-of-plant-seedlings-classification-01-v3-model/20231203_1610_model_history_0.pkl'

## === cell 10
%%time

filename='/kaggle/input/fork-of-plant-seedlings-classification-01-v3-model/20231203_1610_models_0'
best_model = load_model(filename)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/fork-of-plant-seedlings-classification-01-v3-model/20231203_1610_models_0. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/fork-of-plant-seedlings-classification-01-v3-model/20231203_1610_models_0, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 11
%%time

predictions = best_model.predict(test_generator, steps=test_generator.samples)

class_list = []

for i in range(0, predictions.shape[0]):
  y_class = predictions[i, :].argmax(axis=-1)
  class_list += [CLASSES[y_class]]

submission = pd.DataFrame()
submission['file'] = test_generator.filenames
submission['file'] = submission['file'].str.replace(r'test/', '')
submission['species'] = class_list

submission.to_csv('submission.csv', index=False)

print('Submission file generated. All done.')

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'best_model' is not defined
