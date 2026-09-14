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

3.9

# 3. Installed packages



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

0.88512

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

import tensorflow as tf
import tensorflow_hub as hub

from IPython.display import display, Image
from matplotlib.pyplot import imread

from sklearn.model_selection import train_test_split

import os


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("TF version:", tf.__version__)
print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")


## === cell 2
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")


## === cell 3
Labels_Profile = ProfileReport(labels_csv, progress_bar= False)
Labels_Profile.to_notebook_iframe()


## === cell 4
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10));


## === cell 5
filenames = ["../input/dog-breed-identification/train/" + fname + ".jpg" for fname in labels_csv["id"]]

filenames[:5]


## === cell 6
if len(os.listdir("../input/dog-breed-identification/train/")) == len(filenames):
  print("Filenames match actual amount of files!")
else:
  print("Filenames do not match actual amount of files, check the target directory.")


## === cell 7
Image(filenames[42])


## === cell 8
labels = labels_csv["breed"].to_numpy() 
labels[:10]


## === cell 9
if len(labels) == len(filenames):
  print("Number of labels matches number of filenames!")
else:
  print("Number of labels does not match number of filenames, check data directories.")


## === cell 10
unique_breeds = np.unique(labels)
len(unique_breeds)


## === cell 11
boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:2]


## === cell 12
X = filenames
y = boolean_labels


## === cell 13
NUM_IMAGES = 1000 

X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES], 
                                                  test_size=0.2,
                                                  random_state=42)

len(X_train), len(y_train), len(X_val), len(y_val)


## === cell 14
IMG_SIZE = 224

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image


## === cell 15
def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label


## === cell 16
BATCH_SIZE = 32

def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    
    data = data.shuffle(buffer_size=len(x))

    data = data.map(get_image_label)

    data_batch = data.batch(BATCH_SIZE)
  return data_batch


## === cell 17
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


## === cell 18
train_data.element_spec, val_data.element_spec


## === cell 19
def show_25_images(images, labels):
  """
  Displays 25 images from a data batch.
  """
  plt.figure(figsize=(10, 10))
  for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.title(unique_breeds[labels[i].argmax()])
    plt.axis("off")


## === cell 20
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)


## === cell 21
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channel ass

OUTPUT_SHAPE = len(unique_breeds) # number of unique labels

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


## === cell 22
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (the pre-trained model )
    tf.keras.layers.Dense(units=OUTPUT_SHAPE, 
                          activation="softmax") # Layer 2 (output layer)
  ])

  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(), 
      optimizer=tf.keras.optimizers.Adam(), 
      metrics=["accuracy"] 
  )

  model.build(INPUT_SHAPE) 
  
  return model


## === cell 23
model = create_model()
model.summary()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/982339558.py in <cell line: 0>()
      1 # Create a model and check its details
----> 2 model = create_model()
      3 model.summary()

/tmp/ipykernel_11/500143356.py in create_model(input_shape, output_shape, model_url)
      4 
      5   # Setup the model layers using sequential mode.
----> 6   model = tf.keras.Sequential([
      7     hub.KerasLayer(MODEL_URL), # Layer 1 (the pre-trained model )
      8     tf.keras.layers.Dense(units=OUTPUT_SHAPE, 

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7fff0c7d7890> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 24
%load_ext tensorboard


## === cell 25
import datetime

def create_tensorboard_callback():
  logdir = os.path.join("./logs",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)


## === cell 26
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3) # stops after 3 rounds of no improvements


## === cell 27
print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")


## === cell 28
NUM_EPOCHS = 100 
def train_model():
  """
  Trains a given model and returns the trained version.
  """
  model = create_model()

  tensorboard = create_tensorboard_callback()

  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1, # check validation metrics every epoch
            callbacks=[tensorboard, early_stopping])
  
  return model


## === cell 29
model = train_model()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2214319276.py in <cell line: 0>()
      1 # Fit the model to the data
----> 2 model = train_model()

/tmp/ipykernel_11/1837363515.py in train_model()
      6   """
      7   # Create a model
----> 8   model = create_model()
      9 
     10   # Create new TensorBoard session everytime we train a model

/tmp/ipykernel_11/500143356.py in create_model(input_shape, output_shape, model_url)
      4 
      5   # Setup the model layers using sequential mode.
----> 6   model = tf.keras.Sequential([
      7     hub.KerasLayer(MODEL_URL), # Layer 1 (the pre-trained model )
      8     tf.keras.layers.Dense(units=OUTPUT_SHAPE, 

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7fff0c5c4190> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 30
%tensorboard --logdir ./logs 


## === cell 31
predictions = model.predict(val_data, verbose=1) # verbose shows us how long there is to go
predictions


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4189479276.py in <cell line: 0>()
      1 # Make predictions on the validation data (not used to train on)
----> 2 predictions = model.predict(val_data, verbose=1) # verbose shows us how long there is to go
      3 predictions

NameError: name 'model' is not defined

## === cell 32
predictions.shape


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4196273434.py in <cell line: 0>()
      1 # Check the shape of predictions
----> 2 predictions.shape

NameError: name 'predictions' is not defined

## === cell 33
print(predictions[0])
print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model
print(f"Sum: {np.sum(predictions[0])}") # because we used softmax activation in our model, this will be close to 1
print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs
print(f"Predicted label: {unique_breeds[np.argmax(predictions[0])]}") # the predicted label


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/301908975.py in <cell line: 0>()
      1 # First prediction
----> 2 print(predictions[0])
      3 print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model
      4 print(f"Sum: {np.sum(predictions[0])}") # because we used softmax activation in our model, this will be close to 1
      5 print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs

NameError: name 'predictions' is not defined

## === cell 34
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

pred_label = get_pred_label(predictions[0])
pred_label


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2072606372.py in <cell line: 0>()
      7 
      8 # Get a predicted label based on an array of prediction probabilities
----> 9 pred_label = get_pred_label(predictions[0])
     10 pred_label

NameError: name 'predictions' is not defined

## === cell 35
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images, labels

val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]


## === cell 36
def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction, ground truth label and image for sample n.
  """
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
  
  pred_label = get_pred_label(pred_prob)
  
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  plt.title("{} {:2.0f}% ({})".format(pred_label,
                                      np.max(pred_prob)*100,
                                      true_label),
                                      color=color)


## === cell 37
plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3148486062.py in <cell line: 0>()
      1 # View an example prediction, original image and truth label
----> 2 plot_pred(prediction_probabilities=predictions,
      3           labels=val_labels,
      4           images=val_images)

NameError: name 'predictions' is not defined

## === cell 38
def plot_pred_conf(prediction_probabilities, labels, n=1):
  """
  Plots the top 10 highest prediction confidences along with
  the truth label for sample n.
  """
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  pred_label = get_pred_label(pred_prob)

  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                     top_10_pred_values, 
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass


## === cell 39
plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3103730559.py in <cell line: 0>()
----> 1 plot_pred_conf(prediction_probabilities=predictions,
      2                labels=val_labels,
      3                n=9)

NameError: name 'predictions' is not defined

## === cell 40
i_multiplier = 0
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(5*2*num_cols, 5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                labels=val_labels,
                n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/778733439.py in <cell line: 0>()
      7 for i in range(num_images):
      8   plt.subplot(num_rows, 2*num_cols, 2*i+1)
----> 9   plot_pred(prediction_probabilities=predictions,
     10             labels=val_labels,
     11             images=val_images,

NameError: name 'predictions' is not defined

## === cell 41
full_data = create_data_batches(X, y)


## === cell 42
full_model = create_model()


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3467405644.py in <cell line: 0>()
      1 # Instantiate a new model for training on the full dataset
----> 2 full_model = create_model()

/tmp/ipykernel_11/500143356.py in create_model(input_shape, output_shape, model_url)
      4 
      5   # Setup the model layers using sequential mode.
----> 6   model = tf.keras.Sequential([
      7     hub.KerasLayer(MODEL_URL), # Layer 1 (the pre-trained model )
      8     tf.keras.layers.Dense(units=OUTPUT_SHAPE, 

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7fff14248b50> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 43

full_model_tensorboard = create_tensorboard_callback()

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             patience=3)


## === cell 44
%tensorboard --logdir ./logs


## === cell 45
full_model.fit(x=full_data,
               epochs=NUM_EPOCHS,
               callbacks=[full_model_tensorboard, 
                          full_model_early_stopping])


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3236905020.py in <cell line: 0>()
      1 # Fit the full model to the full training data
----> 2 full_model.fit(x=full_data,
      3                epochs=NUM_EPOCHS,
      4                callbacks=[full_model_tensorboard, 
      5                           full_model_early_stopping])

NameError: name 'full_model' is not defined

## === cell 46
def save_model(model, suffix=None):
  """
  Saves a given model in a models directory and appends a suffix (str)
  for clarity and reuse.
  """
  modeldir = os.path.join("./",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
  model_path = modeldir + "-" + suffix + ".h5" # save format of model
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path


## === cell 47
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


## === cell 48
save_model(full_model, suffix="all-images-Adam")


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2979787403.py in <cell line: 0>()
      1 # Save model to file
----> 2 save_model(full_model, suffix="all-images-Adam")

NameError: name 'full_model' is not defined

## === cell 49
loaded_full_model = load_model('./20210103-15251609687518-all-images-Adam.h5')


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3355998900.py in <cell line: 0>()
      1 # Load in the full model
----> 2 loaded_full_model = load_model('./20210103-15251609687518-all-images-Adam.h5')

/tmp/ipykernel_11/3019285423.py in load_model(model_path)
      5   """
      6   print(f"Loading saved model from: {model_path}")
----> 7   model = tf.keras.models.load_model(model_path,
      8                                      custom_objects={"KerasLayer":hub.KerasLayer})
      9   return model

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = './20210103-15251609687518-all-images-Adam.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 50
test_path = "../input/dog-breed-identification/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]

test_filenames[:5]


## === cell 51
len(test_filenames)


## === cell 52
test_data = create_data_batches(test_filenames, test_data=True)


## === cell 53
test_predictions = loaded_full_model.predict(test_data,
                                             verbose=-1)


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/706013120.py in <cell line: 0>()
      1 # Make predictions on test data batch using the loaded full model
----> 2 test_predictions = loaded_full_model.predict(test_data,
      3                                              verbose=-1)

NameError: name 'loaded_full_model' is not defined

## === cell 54
test_predictions[:10]


## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4226608918.py in <cell line: 0>()
      1 # Check out the test predictions
----> 2 test_predictions[:10]

NameError: name 'test_predictions' is not defined

## === cell 55
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()


## === cell 56
test_path = "../input/dog-breed-identification/test/"
preds_df["id"] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df.head()


## === cell 57
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()


## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2495711631.py in <cell line: 0>()
      1 # Add the prediction probabilities to each dog breed column
----> 2 preds_df[list(unique_breeds)] = test_predictions
      3 preds_df.head()

NameError: name 'test_predictions' is not defined

## === cell 58
preds_df.to_csv("./full_submission_1_mobilienetV2_adam.csv",
                 index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission should be the same length as the answers
