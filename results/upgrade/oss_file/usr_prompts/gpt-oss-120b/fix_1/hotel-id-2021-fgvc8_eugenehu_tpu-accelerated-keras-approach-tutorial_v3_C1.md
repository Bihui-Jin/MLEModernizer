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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.0463500884061631

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4

def intialize_accel(hardware):
    """
    input:
    str: GPU or TPU for hardware accelerator
    
    output:
    strategy -- used later for model definition and fitting
    """
    if hardware =='TPU':
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver().connect()  
            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print('TPU Initialized')
            print("TPU Units:", strategy.num_replicas_in_sync)
            return strategy
        except:
            print('TPU Initialization Failed')
    
    
    elif hardware == 'GPU':
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        strategy = tf.distribute.MirroredStrategy()
        return strategy
        
strategy = intialize_accel('TPU')
AUTO = tf.data.experimental.AUTOTUNE


## === cell 6
DIR = '../input/hotel-id-2021-fgvc8/'

Train_PATH = DIR + "/train_images/"
Test_PATH = DIR + "/test_images/"

train_df = pd.read_csv("../input/hotel-id-2021-fgvc8/train.csv")
train_df = train_df.drop_duplicates(subset=['image'])

print("Number of unique hotel chains: ",train_df.chain.nunique())
print("Number of unique hotels: ",train_df.hotel_id.nunique())
print("Number of Training Samples: ", train_df.shape[0])

print(train_df.head())

## === cell 8
Classes =  train_df.hotel_id.nunique()
Channels = 3

size = (200,200)
Split = int(0.9*train_df.shape[0])

## === cell 10
le = preprocessing.LabelEncoder()
train_df['label'] = le.fit_transform(train_df['hotel_id'])
print(train_df[['hotel_id','label']])

## === cell 12
def image_proces(Path, labels):
    """
    inputs: Tensorflow Dataset that contains the following two properties
        Path: Paths to images
        labels: image labels
        
    output: Tensorflow Dataset that contains
        data: processed images
        labels: image labels
    
    Function:
    Takes a tensorflow dataset of image paths and decodes the images into numpy arrays. The images are then resized.
    """
    data = tf.io.read_file(Path)
    data = tf.image.decode_jpeg(data, channels=3)
    data = tf.image.resize(data, size)
    return data,labels

def import_image(Paths,labels):
    """
    inputs:
    Paths: a list of paths to the images
    Y: a list of labels (hotel ids)
    
    outputs:
    dataset: a tensorflow datset that contains images which are decoded and resized
    
    Function: 
    Takes a list of paths and returns a formatted tensorflow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((Paths,labels))
    dataset = dataset.map(image_proces,num_parallel_calls=AUTO)
    return dataset

def data_augment(image, labels):
    """
    inputs: Tensorflow Dataset that contains the following two properties
        Images: Images arrays
        labels: image labels
        
    output: Tensorflow Dataset that contains
        Image: processed images
        labels: image labels
    
    Function:
    Takes a tensorflow dataset of image arrays and augments the brightness and contrast.
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image,lower=0.8,upper=1.2)
    return image, labels


Paths_Test = tf.io.gfile.glob(Test_PATH + '*.jpg')




dataset_Test = import_image(Paths_Test,np.arange(len(Paths_Test)))



## === cell 14
print("Number of Test Samples:",dataset_Test.cardinality().numpy())



## === cell 16


def create_model(Base,input_shape):
    inputs = tf.keras.Input(shape=(input_shape))
    norm =  tf.keras.layers.experimental.preprocessing.Normalization()
    x = norm(inputs)
    x = Base(x,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(50,activation="relu", dtype='float32')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(Classes,activation="softmax", dtype='float32')(x)
        
    model = tf.keras.Model(inputs, outputs)
    return model





## === cell 18
import tensorflow_addons as tfa

def compile_model(model, lr):
    
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,steps_per_execution=8)
    

    return model

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1505785146.py in <cell line: 0>()
----> 1 import tensorflow_addons as tfa
      2 
      3 def compile_model(model, lr):
      4 
      5     optimizer = tf.keras.optimizers.Adam(lr=lr)

ModuleNotFoundError: No module named 'tensorflow_addons'

## === cell 20
EPOCHS= 1
VERBOSE =1




## === cell 22
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                              patience=3, mode='max', min_delta=0.0001,verbose=1)

checkpoint_filepath = './best_model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,verbose=1)


callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max', min_delta=0.0001,verbose=1)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/748314388.py in <cell line: 0>()
      3 
      4 checkpoint_filepath = './best_model.h5'
----> 5 model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      6                             filepath=checkpoint_filepath,
      7                             save_weights_only=True,

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=./best_model.h5

## === cell 24
"""
with strategy.scope():
    Base = tf.keras.applications.ResNet50(include_top=False,input_shape=input_shape)
    Base.trainable = False
    model = create_model(Base,input_shape)
    model = compile_model(model, lr=0.001)
   

print('Fitting') 

History = model.fit(train_dataset, 
                epochs=EPOCHS,
                callbacks=[reduce_lr,model_checkpoint_callback,callback],
                validation_data = val_dataset,  
                verbose=VERBOSE
               )
"""

## === cell 25
"""
from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(History.history['accuracy'][1:],label='Train')
plt.plot(History.history['val_accuracy'][1:],label='Valid')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure(2)
plt.plot(History.history['loss'][1:],label='Train')
plt.plot(History.history['val_loss'][1:],label='Valid')
plt.xlabel('Epoch')
plt.title('Loss')
plt.legend()
plt.show()
"""

## === cell 26
input_shape=[200,200,Channels]
Base = tf.keras.applications.ResNet50(weights=None, include_top=False,input_shape=input_shape)

## === cell 27
checkpoint_filepath = "../input/tpu-accelerated-keras-approach-tutorial/best_model.h5"
Base.trainable = False
best_model = create_model(Base,input_shape)
best_model.load_weights(checkpoint_filepath)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3608030950.py in <cell line: 0>()
      1 checkpoint_filepath = "../input/tpu-accelerated-keras-approach-tutorial/best_model.h5"
      2 Base.trainable = False
----> 3 best_model = create_model(Base,input_shape)
      4 best_model.load_weights(checkpoint_filepath)

/tmp/ipykernel_11/2027460181.py in create_model(Base, input_shape)
      7 def create_model(Base,input_shape):
      8     inputs = tf.keras.Input(shape=(input_shape))
----> 9     norm =  tf.keras.layers.experimental.preprocessing.Normalization()
     10     x = norm(inputs)
     11     x = Base(x,training=False)

AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'experimental'

## === cell 28
dataset_Test = dataset_Test.batch(3)
predictions = best_model.predict(dataset_Test,verbose=1)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/390170886.py in <cell line: 0>()
      1 dataset_Test = dataset_Test.batch(3)
----> 2 predictions = best_model.predict(dataset_Test,verbose=1)

NameError: name 'best_model' is not defined

## === cell 29
images_names= [i.split('/')[-1] for i in Paths_Test]
hotels = le.inverse_transform(np.argmax(predictions,axis=1))
submission = pd.DataFrame(list(zip(images_names,hotels)), columns=['image','hotel_id'])

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1924920345.py in <cell line: 0>()
      1 images_names= [i.split('/')[-1] for i in Paths_Test]
----> 2 hotels = le.inverse_transform(np.argmax(predictions,axis=1))
      3 submission = pd.DataFrame(list(zip(images_names,hotels)), columns=['image','hotel_id'])

NameError: name 'predictions' is not defined

## === cell 30
submission.head()

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

## === cell 31
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
