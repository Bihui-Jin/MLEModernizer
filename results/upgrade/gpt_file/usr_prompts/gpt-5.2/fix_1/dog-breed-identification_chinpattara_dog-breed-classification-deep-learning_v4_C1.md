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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.7623100731269614

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf

print(tf.__version__)

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib, os

## === cell 3
label_df = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
label_df.head()

## === cell 4
label_df.shape

## === cell 5
label_df.breed.value_counts().plot.bar(figsize=(18,5));

## === cell 7
np.max(label_df.breed.value_counts()), np.min(label_df.breed.value_counts())

## === cell 8
label_df.breed.value_counts().plot.hist();

## === cell 9
label_df.breed.value_counts().median()

## === cell 10
im = PIL.Image.open("/kaggle/input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg") 
im

## === cell 12
im = PIL.Image.open(f"/kaggle/input/dog-breed-identification/train/{label_df.id[6]}.jpg") 
im

## === cell 13
img_path=[]
for name in label_df['id']:
    img_path.append(f"/kaggle/input/dog-breed-identification/train/{name}.jpg")
    
    
img_path[:2]

## === cell 14
PIL.Image.open(img_path[9]) # ok got it !!!

## === cell 16
len(os.listdir('/kaggle/input/dog-breed-identification/train')) == len(img_path)

## === cell 17
label_df.shape, len(os.listdir('/kaggle/input/dog-breed-identification/train'))

## === cell 19
labels = label_df.breed.to_numpy()
labels, len(labels)

## === cell 20
len(label_df['breed'].unique())

## === cell 21
label_df.id[60], label_df.breed[60]

## === cell 22
def showImage(path):
    plt.imshow(PIL.Image.open(path))
    
showImage(img_path[60]) 


## === cell 24
np.asarray(PIL.Image.open(img_path[60])).shape 


## === cell 25
samp_i = PIL.Image.open(img_path[60])
print(samp_i.format)
print(samp_i.size) # w (as columns) x h (as rows)

## === cell 28
labels

## === cell 29
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels)

## === cell 30
len(le.classes_)

## === cell 31
label_class = le.classes_

## === cell 32
label_tf = le.transform(labels)
label_tf

## === cell 33
label_class[:5]

## === cell 34
le.inverse_transform([label_tf[0]])

## === cell 35
le.inverse_transform([19])

## === cell 36
label_tf

## === cell 37
label_class[19]

## === cell 38
type(label_tf[0])

## === cell 39
X = img_path
y = label_tf

## === cell 40
len(X), len(y)

## === cell 41
le.inverse_transform([label_tf[1]]), label_df.breed[1]

## === cell 42
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X[:1000],
    y[:1000],
    test_size=0.2,
    random_state=42,
)

len(X_train), len(X_valid)

## === cell 43
X_train[:5], y_train[:5]

## === cell 45
samp = PIL.Image.open(img_path[42])
np.asanyarray(samp).shape # row, col, channel (R,G,B)

## === cell 46
image = imread(img_path[42]) # output np array (257, 350, 3)
plt.imshow(image);

## === cell 49
image.max(), image.min() # 8 bit 0 - 255

## === cell 50
image[:1] # array

## === cell 51
tf.constant(image[:1]) # into tensors

## === cell 52

batch_size = 32
img_height = 224
img_width = 224

## === cell 53

ts = tf.io.read_file(img_path[42])
ts

img = tf.io.decode_jpeg(ts, channels=3)
img


tf.image.resize(img, [img_height, img_width]) # 224x224

## === cell 56
def preprocess_img(sel_path):
    image_raw = tf.io.read_file(sel_path)
    img_dec = tf.io.decode_jpeg(image_raw, channels=3)
    img_nor =tf.image.convert_image_dtype(img_dec, dtype=tf.float32)
    img_re = tf.image.resize(img_nor, [img_height, img_width]) 
    return img_re

## === cell 57
t_img = preprocess_img(img_path[42])
t_img[:2]

## === cell 59
batch_size

## === cell 61
def process_path(file_path, labels):
    label = labels
    img = preprocess_img(file_path)
    return img, label

process_path(X[0], y[0])

## === cell 62
def create_data_batch(X, y=None, valid_data=False, test_data=False):
    if test_data:
        print('creating batchs for testing set...')
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X))
        )
        data_batch = data.map(preprocess_img).batch(batch_size)
        return data_batch
    elif valid_data:
        print('creating batchs for validation set...')
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X), tf.constant(y))
        )
        data_batch = data.map(process_path).batch(batch_size)
        return data_batch
    else:
        print('creating batchs for training set...')
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X), tf.constant(y))
        )
        
        data = data.shuffle(buffer_size=1000) # for training set I'll shuffle
        data = data.map(process_path)
        data_batch = data.batch(batch_size) #return tuple (img, label)
        return data_batch

## === cell 63
train_data = create_data_batch(X_train, y_train)
valid_data = create_data_batch(X_valid, y_valid, valid_data=True)

## === cell 64
train_data.element_spec,valid_data.element_spec

## === cell 67
def batch_img_show(data_set):
    image_batch, label_batch = next(iter(data_set))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i])
        label = label_batch[i]
        l_tf = le.inverse_transform([label])
        plt.title(l_tf[0])
        plt.axis("off")
        
batch_img_show(train_data) # the img will be shuffled when you run it again

## === cell 68
img_batch_sample, label_batch_sample = next(iter(train_data))
len(img_batch_sample), len(label_batch_sample)

## === cell 69
img_batch_sample[0]

## === cell 70
img_batch_sample[:1], label_batch_sample[:1] # (32,32)

## === cell 71
batch_img_show(valid_data) # unlike training set it's not shuffled

## === cell 73
import tensorflow_hub as hub

## === cell 74
img_width, img_height

## === cell 75
INPUT_SHAPE = [None, img_height, img_width, 3]

OUTPUT_SHAPE = len(label_df.breed.unique())

MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5'

## === cell 79
def create_model(model_url):
    print('Building the model ...')
    print('with {}'.format(model_url))
    
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(model_url), # Layer 1
            tf.keras.layers.Dense(units=OUTPUT_SHAPE,activation='softmax'), # Layer 2
            
        ]
    )
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    

    
    model.build(INPUT_SHAPE)
    
    return model

## === cell 80
model = create_model(MODEL_URL)
model.summary()

## --- ERROR in cell 80, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2946360507.py in <cell line: 0>()
----> 1 model = create_model(MODEL_URL)
      2 model.summary()

/tmp/ipykernel_11/3083418292.py in create_model(model_url)
      4 
      5     # setup the model layers
----> 6     model = tf.keras.Sequential(
      7         [
      8             hub.KerasLayer(model_url), # Layer 1

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in __init__(self, layers, trainable, name)
     73         if layers:
     74             for layer in layers:
---> 75                 self.add(layer, rebuild=False)
     76             self._maybe_rebuild()
     77 

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
     95                 layer = origin_layer
     96         if not isinstance(layer, Layer):
---> 97             raise ValueError(
     98                 "Only instances of `keras.Layer` can be "
     99                 f"added to a Sequential model. Received: {layer} "

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f8f1cf55fd0> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 86
import datetime

def create_tensorboard_callback():
    log_dir = "/kaggle/working/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


## === cell 89
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    verbose=1
)

## === cell 90
'Yes' if tf.config.list_physical_devices('GPU') else 'No GPU available'

## === cell 92
tensorboard = create_tensorboard_callback()

model.fit(
    x=train_data,
    validation_data=valid_data,
    epochs=100,
    validation_freq=1,
    callbacks=[tensorboard, early_stopping]
)



## --- ERROR in cell 92, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2326733359.py in <cell line: 0>()
      2 tensorboard = create_tensorboard_callback()
      3 
----> 4 model.fit(
      5     x=train_data,
      6     validation_data=valid_data,

NameError: name 'model' is not defined

## === cell 97
pred = model.predict(valid_data, verbose=1)
pred

## --- ERROR in cell 97, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1752268806.py in <cell line: 0>()
----> 1 pred = model.predict(valid_data, verbose=1)
      2 pred

NameError: name 'model' is not defined

## === cell 98
pred.shape # 120 output -> find highest probability (max)

## --- ERROR in cell 98, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2643647771.py in <cell line: 0>()
----> 1 pred.shape # 120 output -> find highest probability (max)

NameError: name 'pred' is not defined

## === cell 99
i_pred = 101
pd.Series(pred[i_pred]).plot.bar()
plt.xticks([]);

## --- ERROR in cell 99, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1612229373.py in <cell line: 0>()
      1 i_pred = 101
----> 2 pd.Series(pred[i_pred]).plot.bar()
      3 plt.xticks([]);

NameError: name 'pred' is not defined

## === cell 100
pred[i_pred].sum() # sum of softmax apx 1 (0, 1)

## --- ERROR in cell 100, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/679919772.py in <cell line: 0>()
----> 1 pred[i_pred].sum() # sum of softmax apx 1 (0, 1)

NameError: name 'pred' is not defined

## === cell 101
np.max(pred[i_pred])

## --- ERROR in cell 101, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2121171587.py in <cell line: 0>()
      1 # high probability
----> 2 np.max(pred[i_pred])

NameError: name 'pred' is not defined

## === cell 102
np.argmax(pred[i_pred])

## --- ERROR in cell 102, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/177201543.py in <cell line: 0>()
----> 1 np.argmax(pred[i_pred])

NameError: name 'pred' is not defined

## === cell 103
le.inverse_transform([np.argmax(pred[i_pred])])

## --- ERROR in cell 103, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3584689607.py in <cell line: 0>()
----> 1 le.inverse_transform([np.argmax(pred[i_pred])])

NameError: name 'pred' is not defined

## === cell 105
np.argmax(pred[i_pred]), y_valid[i_pred]

## --- ERROR in cell 105, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2229819593.py in <cell line: 0>()
----> 1 np.argmax(pred[i_pred]), y_valid[i_pred]

NameError: name 'pred' is not defined

## === cell 106
un_b = valid_data.unbatch()
list(un_b.as_numpy_iterator())[:1]

## === cell 108
unBatch_images = []
unBatch_labels = []

for image, label in un_b.as_numpy_iterator():
    unBatch_images.append(image)
    unBatch_labels.append(label)

## === cell 111
unBatch_images[5].dtype

## === cell 113
pred_label = le.inverse_transform([np.argmax(pred[i_pred])])[0]

plt.imshow(unBatch_images[i_pred])
plt.title(f'Actual: {le.inverse_transform([unBatch_labels[i_pred]])[0]}\nPredict: ({np.max(pred[i_pred])*100:.2f} %): {pred_label})')
plt.axis('off');

## --- ERROR in cell 113, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2213042055.py in <cell line: 0>()
----> 1 pred_label = le.inverse_transform([np.argmax(pred[i_pred])])[0]
      2 
      3 plt.imshow(unBatch_images[i_pred])
      4 plt.title(f'Actual: {le.inverse_transform([unBatch_labels[i_pred]])[0]}\nPredict: ({np.max(pred[i_pred])*100:.2f} %): {pred_label})')
      5 plt.axis('off');

NameError: name 'pred' is not defined

## === cell 114
s_pred = pred

## --- ERROR in cell 114, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379810214.py in <cell line: 0>()
----> 1 s_pred = pred

NameError: name 'pred' is not defined

## === cell 115
idx_sort = np.argsort(s_pred[i_pred])
idx_sort = idx_sort[::-1][:5]
idx_sort

## --- ERROR in cell 115, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3127427582.py in <cell line: 0>()
      1 # sort (ascending) and return index
----> 2 idx_sort = np.argsort(s_pred[i_pred])
      3 idx_sort = idx_sort[::-1][:5]
      4 idx_sort

NameError: name 's_pred' is not defined

## === cell 116
s_pred[i_pred][idx_sort]

## --- ERROR in cell 116, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4119204002.py in <cell line: 0>()
----> 1 s_pred[i_pred][idx_sort]

NameError: name 's_pred' is not defined

## === cell 117
top_5_prob = np.sort(s_pred[i_pred]*100)[::-1][:5]
top_5_prob

## --- ERROR in cell 117, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3812675820.py in <cell line: 0>()
      1 # sort value (ascending)
----> 2 top_5_prob = np.sort(s_pred[i_pred]*100)[::-1][:5]
      3 top_5_prob

NameError: name 's_pred' is not defined

## === cell 118
pred_label_top5 = []
for name in idx_sort:
    inv = le.inverse_transform([name])[0]
    pred_label_top5.append(inv)
    
pred_label_top5

## --- ERROR in cell 118, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2867012423.py in <cell line: 0>()
      1 pred_label_top5 = []
----> 2 for name in idx_sort:
      3     inv = le.inverse_transform([name])[0]
      4     pred_label_top5.append(inv)
      5 

NameError: name 'idx_sort' is not defined

## === cell 119
plt.bar(pred_label_top5, top_5_prob)
plt.title('Top 5 probability prediction')
plt.xticks(rotation=45);

## --- ERROR in cell 119, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3400758642.py in <cell line: 0>()
----> 1 plt.bar(pred_label_top5, top_5_prob)
      2 plt.title('Top 5 probability prediction')
      3 plt.xticks(rotation=45);

NameError: name 'top_5_prob' is not defined

## === cell 120
top5_img = []
top5_label = []

for img, label in zip(X[:1000], y[:1000]):
    for l in idx_sort:
        if label == l and label not in top5_label:
            top5_img.append(
                preprocess_img(img)
            )
            top5_label.append(label)

## --- ERROR in cell 120, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2349539280.py in <cell line: 0>()
      3 
      4 for img, label in zip(X[:1000], y[:1000]):
----> 5     for l in idx_sort:
      6         if label == l and label not in top5_label:
      7             top5_img.append(

NameError: name 'idx_sort' is not defined

## === cell 121
idx_sort

## --- ERROR in cell 121, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/991739612.py in <cell line: 0>()
----> 1 idx_sort

NameError: name 'idx_sort' is not defined

## === cell 122
top5_label

## === cell 123
sorted_img = []
sorted_label = []
for c in range(len(idx_sort)):
    print(np.where(idx_sort[c] == top5_label))
    new_idx = np.where(idx_sort[c] == top5_label)[0][0]
    sorted_img.append(top5_img[new_idx])
    sorted_label.append(top5_label[new_idx])

## --- ERROR in cell 123, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3622635352.py in <cell line: 0>()
      2 sorted_label = []
      3 # Dynamic sort (for image that using label tag to sort)
----> 4 for c in range(len(idx_sort)):
      5     print(np.where(idx_sort[c] == top5_label))
      6     new_idx = np.where(idx_sort[c] == top5_label)[0][0]

NameError: name 'idx_sort' is not defined

## === cell 124
sorted_label == idx_sort

## --- ERROR in cell 124, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2083485515.py in <cell line: 0>()
----> 1 sorted_label == idx_sort

NameError: name 'idx_sort' is not defined

## === cell 126
plt.figure(figsize=(10,10))
plt.suptitle('TOP 5 prediction')
for i, label in enumerate(idx_sort):
    ax = plt.subplot(3,2, i+1)
    plt.imshow(sorted_img[i])
    plt.title('{} ({:.2f} %)'.format(le.inverse_transform([label])[0], top_5_prob[i]))
    plt.axis('off')

## --- ERROR in cell 126, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3515358204.py in <cell line: 0>()
      1 plt.figure(figsize=(10,10))
      2 plt.suptitle('TOP 5 prediction')
----> 3 for i, label in enumerate(idx_sort):
      4     ax = plt.subplot(3,2, i+1)
      5     plt.imshow(sorted_img[i])

NameError: name 'idx_sort' is not defined

## === cell 127
model.evaluate(valid_data)

## --- ERROR in cell 127, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/512471090.py in <cell line: 0>()
      1 # Evaluating -> this model's only trained on 1000 datasets
----> 2 model.evaluate(valid_data)

NameError: name 'model' is not defined

## === cell 129
def save_model(model, suffix=None):
    model_dir = os.path.join(
        '/kaggle/working/models',
        datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    )
    model_path = model_dir + '-' + suffix + '.h5'
    print(f'Saving model to: {model_path}')
    model.save(model_path)
    return model_path

## === cell 130
def load_model(model_path):
    print(f'Loading model from: "{model_path}"')
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'KerasLayer': hub.KerasLayer
        }
    )
    return model

## === cell 137
print(f'Full datasets X,y = {len(X)}, {len(y)}')
print(f'test datasets X`,y` = {len(X_train) + len(X_valid)}, {len(y_train) + len(y_valid)}')

## === cell 138
full_data_batch = create_data_batch(X, y)

## === cell 139
full_model = create_model(MODEL_URL)

## --- ERROR in cell 139, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3126591753.py in <cell line: 0>()
----> 1 full_model = create_model(MODEL_URL)

/tmp/ipykernel_11/3083418292.py in create_model(model_url)
      4 
      5     # setup the model layers
----> 6     model = tf.keras.Sequential(
      7         [
      8             hub.KerasLayer(model_url), # Layer 1

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in __init__(self, layers, trainable, name)
     73         if layers:
     74             for layer in layers:
---> 75                 self.add(layer, rebuild=False)
     76             self._maybe_rebuild()
     77 

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
     95                 layer = origin_layer
     96         if not isinstance(layer, Layer):
---> 97             raise ValueError(
     98                 "Only instances of `keras.Layer` can be "
     99                 f"added to a Sequential model. Received: {layer} "

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f8ee0cc6b10> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 140
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=3
)

## === cell 143
dog_breed_clf_model = load_model('/kaggle/input/dog-clf-model/20230615-13451686836717-FULL-1000-img-mobilenetV2-OpAdam.h5')

## --- ERROR in cell 143, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3077514440.py in <cell line: 0>()
      1 # Load pretained model (FUll)
----> 2 dog_breed_clf_model = load_model('/kaggle/input/dog-clf-model/20230615-13451686836717-FULL-1000-img-mobilenetV2-OpAdam.h5')

/tmp/ipykernel_11/2357183805.py in load_model(model_path)
      1 def load_model(model_path):
      2     print(f'Loading model from: "{model_path}"')
----> 3     model = tf.keras.models.load_model(
      4         model_path,
      5         custom_objects={

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/dog-clf-model/20230615-13451686836717-FULL-1000-img-mobilenetV2-OpAdam.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 145
test_path = '/kaggle/input/dog-breed-identification/test'

## === cell 146
dir_list = os.listdir(test_path)
file_name = [os.path.splitext(name)[0] for name in dir_list]
print(len(file_name))


## === cell 147
sample_id = 499
plt.imshow(imread(f'{test_path}/{dir_list[sample_id]}'))
plt.axis('off');

## === cell 148
preds_submit = pd.DataFrame(file_name, columns=['id'])
preds_submit.head()

## === cell 149
test_img_path = []
for name in dir_list:
    path = test_path + '/' + name
    test_img_path.append(path)
    
print(len(test_img_path))
test_img_path[:2]

## === cell 150
plt.imshow(preprocess_img(test_img_path[678]));

## === cell 151
test_data = create_data_batch(test_img_path, test_data=True)
test_data

## === cell 153
unique_lb = label_df.breed.unique()
sort_label = np.sort(unique_lb)
print(len(sort_label))
sort_label[:5]

## === cell 154
y_pred = dog_breed_clf_model.predict(test_data)

## --- ERROR in cell 154, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2610594117.py in <cell line: 0>()
----> 1 y_pred = dog_breed_clf_model.predict(test_data)

NameError: name 'dog_breed_clf_model' is not defined

## === cell 156
test_samp_i = 7
np.argmax(y_pred[test_samp_i])

## --- ERROR in cell 156, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2534380854.py in <cell line: 0>()
      1 # find index of class that having highest probability
      2 test_samp_i = 7
----> 3 np.argmax(y_pred[test_samp_i])

NameError: name 'y_pred' is not defined

## === cell 157
le.inverse_transform([np.argmax(y_pred[test_samp_i])])[0]

## --- ERROR in cell 157, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006320998.py in <cell line: 0>()
----> 1 le.inverse_transform([np.argmax(y_pred[test_samp_i])])[0]

NameError: name 'y_pred' is not defined

## === cell 158
pd.Series(y_pred[test_samp_i]).plot.bar()
plt.xticks([]);

## --- ERROR in cell 158, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/946126993.py in <cell line: 0>()
----> 1 pd.Series(y_pred[test_samp_i]).plot.bar()
      2 plt.xticks([]);

NameError: name 'y_pred' is not defined

## === cell 159
plt.imshow(imread(test_img_path[test_samp_i]))
plt.title(f'pred: {le.inverse_transform([np.argmax(y_pred[test_samp_i])])[0]} ({y_pred[test_samp_i][np.argmax(y_pred[test_samp_i])]*100:.2f} %)')
plt.axis('off');

## --- ERROR in cell 159, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4083774779.py in <cell line: 0>()
      1 plt.imshow(imread(test_img_path[test_samp_i]))
----> 2 plt.title(f'pred: {le.inverse_transform([np.argmax(y_pred[test_samp_i])])[0]} ({y_pred[test_samp_i][np.argmax(y_pred[test_samp_i])]*100:.2f} %)')
      3 plt.axis('off');

NameError: name 'y_pred' is not defined

## === cell 160
breed_pred = pd.DataFrame(y_pred, columns=le.classes_)
breed_pred

## --- ERROR in cell 160, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3590149502.py in <cell line: 0>()
----> 1 breed_pred = pd.DataFrame(y_pred, columns=le.classes_)
      2 breed_pred

NameError: name 'y_pred' is not defined

## === cell 161
preds_submit = preds_submit.join(breed_pred)

## --- ERROR in cell 161, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1900748795.py in <cell line: 0>()
      1 # Filled the submit dataframe
----> 2 preds_submit = preds_submit.join(breed_pred)

NameError: name 'breed_pred' is not defined

## === cell 162
preds_submit.shape

## === cell 163
preds_submit.head()

## === cell 164
preds_submit.to_csv('submission.csv', index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have columns for all dogs: ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']
