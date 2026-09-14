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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
plotly==5.24.1
plotly-express==0.4.1
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.75599

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

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
import numpy as np
import sys
import tensorflow as tf
import os
import sys
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


%matplotlib inline
import matplotlib.image as img
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import itertools
import plotly.express as px


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


## === cell 3
import zipfile

zip_files = ['test', 'train']

for zip_file in zip_files:
    with zipfile.ZipFile("../input/dogs-vs-cats-redux-kernels-edition/{}.zip".format(zip_file),"r") as z:
        z.extractall(".")
        print("{} unzipped".format(zip_file))


## === cell 4
print(os.listdir('../working'))


## === cell 5
IMAGE_FOLDER_PATH = "../working/train"
FILE_NAMES = os.listdir(IMAGE_FOLDER_PATH)
WIDTH = 150
HEIGHT = 150


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1174165896.py in <cell line: 0>()
      1 IMAGE_FOLDER_PATH = "../working/train"
----> 2 FILE_NAMES = os.listdir(IMAGE_FOLDER_PATH)
      3 WIDTH = 150
      4 HEIGHT = 150

FileNotFoundError: [Errno 2] No such file or directory: '../working/train'

## === cell 6
FILE_NAMES[0:5]


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3406717083.py in <cell line: 0>()
----> 1 FILE_NAMES[0:5]

NameError: name 'FILE_NAMES' is not defined

## === cell 7
labels = []
for i in os.listdir(IMAGE_FOLDER_PATH):
    labels+=[i]


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/324667010.py in <cell line: 0>()
      1 labels = []
----> 2 for i in os.listdir(IMAGE_FOLDER_PATH):
      3     labels+=[i]

FileNotFoundError: [Errno 2] No such file or directory: '../working/train'

## === cell 8
targets = list()
full_paths = list()
train_cats_dir = list()
train_dogs_dir = list()

for file_name in FILE_NAMES:
    target = file_name.split(".")[0] # target name
    full_path = os.path.join(IMAGE_FOLDER_PATH, file_name)
    
    if(target == "dog"):
        train_dogs_dir.append(full_path)
    if(target == "cat"):
        train_cats_dir.append(full_path)
    
    full_paths.append(full_path)
    targets.append(target)

dataset = pd.DataFrame() # make dataframe
dataset['image_path'] = full_paths # file path
dataset['target'] = targets # file's target


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3711016638.py in <cell line: 0>()
      6 
      7 # finding each file's target
----> 8 for file_name in FILE_NAMES:
      9     target = file_name.split(".")[0] # target name
     10     full_path = os.path.join(IMAGE_FOLDER_PATH, file_name)

NameError: name 'FILE_NAMES' is not defined

## === cell 9
dataset.head(10)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3313169758.py in <cell line: 0>()
----> 1 dataset.head(10)

NameError: name 'dataset' is not defined

## === cell 10
print("total data counts:", dataset['target'].count())
counts = dataset['target'].value_counts()
print(counts)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2716419378.py in <cell line: 0>()
----> 1 print("total data counts:", dataset['target'].count())
      2 counts = dataset['target'].value_counts()
      3 print(counts)

NameError: name 'dataset' is not defined

## === cell 11
dataset_train, dataset_test = train_test_split(dataset, test_size=0.5, random_state=seed)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2951695304.py in <cell line: 0>()
----> 1 dataset_train, dataset_test = train_test_split(dataset, test_size=0.5, random_state=seed)

NameError: name 'dataset' is not defined

## === cell 12
class_id_distributionTrain = dataset_train['target'].value_counts()
class_id_distributionTrain.head(10)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/370431940.py in <cell line: 0>()
----> 1 class_id_distributionTrain = dataset_train['target'].value_counts()
      2 class_id_distributionTrain.head(10)

NameError: name 'dataset_train' is not defined

## === cell 13
class_id_distributionTest = dataset_test['target'].value_counts()
class_id_distributionTest.head(10)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3516145187.py in <cell line: 0>()
----> 1 class_id_distributionTest = dataset_test['target'].value_counts()
      2 class_id_distributionTest.head(10)

NameError: name 'dataset_test' is not defined

## === cell 14
train_datagen=ImageDataGenerator(
rotation_range=15,
rescale=1./255,
shear_range=0.1,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1)

train_datagenerator=train_datagen.flow_from_dataframe(dataframe=dataset_train,
                                                     x_col="image_path",
                                                     y_col="target",
                                                     target_size=(WIDTH, HEIGHT),
                                                     class_mode="binary",
                                                     batch_size=150)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1536948121.py in <cell line: 0>()
      8 height_shift_range=0.1)
      9 
---> 10 train_datagenerator=train_datagen.flow_from_dataframe(dataframe=dataset_train,
     11                                                      x_col="image_path",
     12                                                      y_col="target",

NameError: name 'dataset_train' is not defined

## === cell 15
test_datagen = ImageDataGenerator(rescale=1./255)
test_datagenerator=test_datagen.flow_from_dataframe(dataframe=dataset_test,
                                                   x_col="image_path",
                                                   y_col="target",
                                                   target_size=(WIDTH, HEIGHT),
                                                   class_mode="binary",
                                                   batch_size=150)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2423752633.py in <cell line: 0>()
      1 test_datagen = ImageDataGenerator(rescale=1./255)
----> 2 test_datagenerator=test_datagen.flow_from_dataframe(dataframe=dataset_test,
      3                                                    x_col="image_path",
      4                                                    y_col="target",
      5                                                    target_size=(WIDTH, HEIGHT),

NameError: name 'dataset_test' is not defined

## === cell 16
model = Sequential() # implement model layer 
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1784914027.py in <cell line: 0>()
      1 model = Sequential() # implement model layer
----> 2 model.add(Conv2D(32, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), activation='relu'))
      3 model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
      4 model.add(MaxPooling2D(pool_size=2))
      5 model.add(Dropout(0.25))

NameError: name 'WIDTH' is not defined

## === cell 17
from tensorflow.keras.utils import plot_model
from IPython.display import Image
plot_model(model, to_file='convnet.png', show_shapes=True,show_layer_names=True)
Image(filename='convnet.png') 


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1247658202.py in <cell line: 0>()
      1 from tensorflow.keras.utils import plot_model
      2 from IPython.display import Image
----> 3 plot_model(model, to_file='convnet.png', show_shapes=True,show_layer_names=True)
      4 Image(filename='convnet.png')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/model_visualization.py in plot_model(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, show_layer_activations, show_trainable, **kwargs)
    418 
    419     if not model.built:
--> 420         raise ValueError(
    421             "This model has not yet been built. "
    422             "Build the model first by calling `build()` or by calling "

ValueError: This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.

## === cell 18
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.summary()


## === cell 19
History=model.fit(train_datagenerator,
                       epochs=1,
                       validation_data=test_datagenerator,
                       validation_steps=dataset_test.shape[0]/150,
                       steps_per_epoch=dataset_train.shape[0]/150)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1332139683.py in <cell line: 0>()
----> 1 History=model.fit(train_datagenerator,
      2                        epochs=1,
      3                        validation_data=test_datagenerator,
      4                        validation_steps=dataset_test.shape[0]/150,
      5                        steps_per_epoch=dataset_train.shape[0]/150)

NameError: name 'train_datagenerator' is not defined

## === cell 20
acc = History.history['accuracy']
val_acc = History.history['val_accuracy']
loss = History.history['loss']
val_loss = History.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2064919136.py in <cell line: 0>()
----> 1 acc = History.history['accuracy']
      2 val_acc = History.history['val_accuracy']
      3 loss = History.history['loss']
      4 val_loss = History.history['val_loss']
      5 

NameError: name 'History' is not defined

## === cell 21
test_loss, test_acc = model.evaluate(test_datagenerator, steps=len(test_datagenerator), verbose=1)
print('Loss: %.3f' % (test_loss * 100.0)) 
print('Accuracy: %.3f' % (test_acc * 100.0))


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/576976338.py in <cell line: 0>()
----> 1 test_loss, test_acc = model.evaluate(test_datagenerator, steps=len(test_datagenerator), verbose=1)
      2 print('Loss: %.3f' % (test_loss * 100.0))
      3 print('Accuracy: %.3f' % (test_acc * 100.0))

NameError: name 'test_datagenerator' is not defined

## === cell 22
from sklearn.metrics import confusion_matrix
import itertools


## === cell 23
predictions = model.predict(x=test_datagenerator, steps= len(test_datagenerator), verbose=0)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3845067836.py in <cell line: 0>()
----> 1 predictions = model.predict(x=test_datagenerator, steps= len(test_datagenerator), verbose=0)

NameError: name 'test_datagenerator' is not defined

## === cell 24
test_datagenerator.classes


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2409482587.py in <cell line: 0>()
----> 1 test_datagenerator.classes

NameError: name 'test_datagenerator' is not defined

## === cell 25
cm = confusion_matrix(y_true=test_datagenerator.classes, y_pred=np.argmax(predictions, axis=-1))


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3565314356.py in <cell line: 0>()
----> 1 cm = confusion_matrix(y_true=test_datagenerator.classes, y_pred=np.argmax(predictions, axis=-1))

NameError: name 'test_datagenerator' is not defined

## === cell 26
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


## === cell 27
cm_plot_labels = ['0_cat', '1_dog']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3475869663.py in <cell line: 0>()
      1 cm_plot_labels = ['0_cat', '1_dog']
      2 
----> 3 plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

NameError: name 'cm' is not defined

## === cell 28
def gen_image_label(directory):
    ''' A generator that yields (label, id, jpg_filename) tuple.'''
    for root, dirs, files in os.walk(directory):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext != '.jpg':
                continue
            basename = os.path.basename(f)
            splits = basename.split('.')
            if len(splits) == 3:
                label, id_, ext = splits
            else:
                label = None
                id_, ext = splits
            fullname = os.path.join(root, f)
            yield label, int(id_), fullname


## === cell 29
test_data_dir = "../working/test/"
lst = list(gen_image_label(test_data_dir))
test_df = pd.DataFrame(lst, columns=['label', 'id', 'filename'])
test_df = test_df.sort_values(by=['label', 'id'])
test_df['label_code'] = test_df.label.map({'cat':0, 'dog':1})

test_df.head(3)


## === cell 30
predictions.shape


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2983742165.py in <cell line: 0>()
----> 1 predictions.shape

NameError: name 'predictions' is not defined

## === cell 31
results = pd.DataFrame({'id': pd.Series(test_df.id.values[:predictions.shape[0]]),
                        'label': pd.Series(predictions.T[0])})
results.to_csv('submission.csv', index=False)
results.head(10)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1960817081.py in <cell line: 0>()
      1 # csvの作成
----> 2 results = pd.DataFrame({'id': pd.Series(test_df.id.values[:predictions.shape[0]]),
      3                         'label': pd.Series(predictions.T[0])})
      4 results.to_csv('submission.csv', index=False)
      5 results.head(10)

NameError: name 'predictions' is not defined
