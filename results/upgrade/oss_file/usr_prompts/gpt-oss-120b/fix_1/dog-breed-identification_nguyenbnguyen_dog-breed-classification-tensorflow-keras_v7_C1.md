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

3.13

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

1.581792533337866

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)

KAGGLE_DATA_DIR = Path("/kaggle/input/dog-breed-identification")

labels_df = pd.read_csv(KAGGLE_DATA_DIR / 'labels.csv')

filenames = [str(KAGGLE_DATA_DIR / f'train/{filename}.jpg') for filename in labels_df["id"]]

## === cell 3
labels = labels_df["breed"].to_numpy()
len(labels) == len(filenames)

## === cell 4
filenames[:5]

## === cell 5
len(filenames)

## === cell 7
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import Image

%matplotlib inline

## === cell 8
unique_breeds = np.unique(labels)
unique_breeds[:10]

## === cell 9
labels_df.head()

## === cell 10
labels_df.describe()

## === cell 11
labels_df["breed"].value_counts(ascending=True).plot.barh(figsize=(20, 30))
plt.tight_layout()
plt.show()

## === cell 12
example_dog_breed_name = labels_df[labels_df["id"] == '0021f9ceb3235effd7fcde7f7538ed62']['breed'].values[0]

print(f"{example_dog_breed_name}")

example_dog_breed = Image(KAGGLE_DATA_DIR /'train/0021f9ceb3235effd7fcde7f7538ed62.jpg')
example_dog_breed

## === cell 13
image = plt.imread(filenames[0])
image.shape

## === cell 14
image[:1]

## === cell 16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

IMG_WIDTH = 224
IMG_HEIGHT = IMG_WIDTH
IMG_CHANNELS = 3

BATCH_SIZE = 32

@tf.autograph.experimental.do_not_convert
def process_image(image_path: str):
    """
    Take an image file path and turns the image into a Tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_WIDTH, IMG_HEIGHT])
    return image

@tf.autograph.experimental.do_not_convert
def get_image_label(image_path: str, label: str):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple (image, label)
    """
    image = process_image(image_path)
    return image, label

def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
        return data_batch

    if valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((
                                                  tf.constant(X), # filepaths
                                                  tf.constant(y)  # labels
                                                  ))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
        return data_batch

    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((
                                             tf.constant(X),
                                             tf.constant(y)
                                            ))
    data_batch = data.shuffle(buffer_size=len(X)) \
                   .map(get_image_label) \
                   .batch(BATCH_SIZE) \
                   .prefetch(tf.data.AUTOTUNE).cache()
    return data_batch

random_state = 42
random_seed = random_state

tf.random.set_seed(random_seed)

print("TF version:", tf.__version__)

if tf.config.list_physical_devices("GPU"):
    print("GPU enabled")
else:
    print("GPU is not available, switch to CPU")

lb = LabelBinarizer()

encoded_labels = lb.fit_transform(labels)

print(f"{encoded_labels[:1] = }")

X_train, X_test, y_train, y_test = train_test_split(filenames, 
                                                    encoded_labels, 
                                                    test_size=0.1, 
                                                    random_state=random_state, 
                                                    stratify=encoded_labels)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                      y_train, 
                                                      test_size=0.1, 
                                                      random_state=7, 
                                                      stratify=y_train)

train_data = create_data_batches(X_train, y_train)
valid_data = create_data_batches(X_valid, y_valid, valid_data=True)
test_data_ = create_data_batches(X_test, y_test, valid_data=True)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
train_data.element_spec, valid_data.element_spec

## === cell 18
def show_25_images(images, labels):
    """
    Displays a plot of 25 images and their labels from a data batch.
    """
    plt.figure(figsize=(15, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(images[i])
        plt.title(unique_breeds[np.argmax(labels[i])])
        plt.axis("off")
    plt.tight_layout()

## === cell 19
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)

## === cell 20
valid_images, valid_labels = next(valid_data.as_numpy_iterator())
show_25_images(valid_images, valid_labels)

## === cell 21
test_images, test_labels = next(test_data_.as_numpy_iterator())
show_25_images(test_images, test_labels)

## === cell 23
import tensorflow_hub as hub
from tensorflow.keras import layers


def save_model(model, file_name, path_folder='./', include_datetime=True):
    """
    Saves a given model in a models directory and appends a suffix (string).
    """
    suffix = None
    
    if include_datetime:
        suffix = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%s")}'
    
    if suffix:
        full_file_name = f'{suffix}-{file_name}.keras'
    else:
        full_file_name = f'{file_name}.keras'
    
    model_path = Path(path_folder) / full_file_name
    
    print(f"Saving model to: {model_path}...")
    
    model.save(model_path)
    
    return model_path

def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    
    print(f"Loading saved model from: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    return model
    
def MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), 
                num_classes=len(unique_breeds), 
                base_model_trainable = False,
                data_augmentation=None):
    
    hub_layer = hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-224-classification/2",
               trainable=base_model_trainable)
    
    hub_layer.build(input_shape)
    
    inputs = layers.Input(input_shape)
    

    X = inputs
    
    X = hub_layer(X)

    
    outputs = layers.Dense(num_classes)(X)
    
    return tf.keras.Model(inputs = inputs, outputs=outputs)


def create_model(model=MobileNetV2(), learning_rate=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tf.keras.models.Sequential()
    data_augmentation.add(layers.RandomFlip('horizontal'))
    data_augmentation.add(layers.RandomRotation(0.2))
    
    return data_augmentation

lr = 1e-2


model = create_model(learning_rate=lr)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=.3, 
    patience=2, 
    min_lr=1e-7
)

val_acc_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    restore_best_weights=True, 
    patience=10,
    start_from_epoch = 5,
)

model.summary(show_trainable=True)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3283002042.py in <cell line: 0>()
     66 
     67 
---> 68 def create_model(model=MobileNetV2(), learning_rate=1e-3):
     69     model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
     70                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

/tmp/ipykernel_11/3283002042.py in MobileNetV2(input_shape, num_classes, base_model_trainable, data_augmentation)
     57     X = inputs
     58 
---> 59     X = hub_layer(X)
     60 
     61     # X = layers.Dropout(.5)(X)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    248         # Behave like BatchNormalization. (Dropout is different, b/181839368.)
    249         training = False
--> 250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
    252                                      lambda: f(training=False))

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in <lambda>()
    250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
--> 252                                      lambda: f(training=False))
    253 
    254     # Unwrap dicts returned by signatures.

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in canonicalize_to_monomorphic(args, kwargs, default_values, capture_types, polymorphic_type)
    581     else:
    582       parameters.append(
--> 583           _make_validated_mono_param(name, arg, poly_parameter.kind,
    584                                      type_context,
    585                                      poly_parameter.type_constraint))

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in _make_validated_mono_param(name, value, kind, type_context, poly_type)
    520 ) -> Parameter:
    521   """Generates and validates a parameter for Monomorphic FunctionType."""
--> 522   mono_type = trace_type.from_value(value, type_context)
    523 
    524   if poly_type and not mono_type.is_subtype_of(poly_type):

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py in from_value(value, context)
    183 
    184   if util.is_np_ndarray(value):
--> 185     ndarray = value.__array__()
    186     return default_types.TENSOR(ndarray.shape, ndarray.dtype)
    187 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __array__(self)
    106 
    107     def __array__(self):
--> 108         raise ValueError(
    109             "A KerasTensor is symbolic: it's a placeholder for a shape "
    110             "an a dtype. It doesn't have any actual numerical value. "

ValueError: Exception encountered when calling layer 'keras_layer' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'keras_layer' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=keras_tensor>
  • training=None

## === cell 26
epochs = 20

history = model.fit(train_data, 
                    validation_data=valid_data, 
                    callbacks=[reduce_lr, val_acc_stopping],
                    epochs=epochs)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/436483159.py in <cell line: 0>()
      1 epochs = 20
      2 
----> 3 history = model.fit(train_data, 
      4                     validation_data=valid_data,
      5                     callbacks=[reduce_lr, val_acc_stopping],

NameError: name 'model' is not defined

## === cell 28
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3483149904.py in <cell line: 0>()
----> 1 acc = [0.] + history.history['accuracy']
      2 val_acc = [0.] + history.history['val_accuracy']
      3 
      4 loss = history.history['loss']
      5 val_loss = history.history['val_loss']

NameError: name 'history' is not defined

## === cell 29
base_model = model.layers[-2]
base_model.trainable = True

model.summary(show_trainable=True)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4164868213.py in <cell line: 0>()
----> 1 base_model = model.layers[-2]
      2 base_model.trainable = True
      3 
      4 model.summary(show_trainable=True)

NameError: name 'model' is not defined

## === cell 30
loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(lr * 1e-3)
metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=np.float32)]

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3276610061.py in <cell line: 0>()
      1 loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
----> 2 optimizer = tf.keras.optimizers.Adam(lr * 1e-3)
      3 metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=np.float32)]

NameError: name 'lr' is not defined

## === cell 31
fine_tune_epochs = 50
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(train_data, 
                         validation_data=valid_data, 
                         epochs=total_epochs, 
                         callbacks=[reduce_lr, val_acc_stopping],
                         initial_epoch=history.epoch[-1])

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/273229348.py in <cell line: 0>()
      2 total_epochs = epochs + fine_tune_epochs
      3 
----> 4 history_fine = model.fit(train_data, 
      5                          validation_data=valid_data,
      6                          epochs=total_epochs,

NameError: name 'model' is not defined

## === cell 32
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2756751752.py in <cell line: 0>()
----> 1 acc += history_fine.history['accuracy']
      2 val_acc += history_fine.history['val_accuracy']
      3 
      4 loss += history_fine.history['loss']
      5 val_loss += history_fine.history['val_loss']

NameError: name 'acc' is not defined

## === cell 33
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([epochs-1,epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([epochs-1,epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3020243951.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 8))
      2 plt.subplot(2, 1, 1)
----> 3 plt.plot(acc, label='Training Accuracy')
      4 plt.plot(val_acc, label='Validation Accuracy')
      5 plt.ylim([0, 1])

NameError: name 'acc' is not defined

## === cell 34
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]
    
preds = model.predict(test_data_, verbose=0)

index = 0
print(f"Max value (probability of prediction): {np.max(tf.nn.softmax(preds[index]))}")
print("Sum:", np.sum(tf.nn.softmax(preds[index])))
print("Max index:", np.argmax(preds[index]))
print("Predicted label:", unique_breeds[np.argmax(preds[index])])
print("Actual label:", unique_breeds[np.argmax(y_test[index])])

pred_label = get_pred_label(preds[7])
print(f"{pred_label = }")

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3854482760.py in <cell line: 0>()
      5   return unique_breeds[np.argmax(prediction_probabilities)]
      6 
----> 7 preds = model.predict(test_data_, verbose=0)
      8 
      9 # First prediction

NameError: name 'model' is not defined

## === cell 35
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and reutrns separate arrays
    of images and labels.
    """
    images = []
    labels = []
    
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_breeds[np.argmax(label)])
        
    return images, labels

test_images, test_labels = unbatchify(test_data_)
test_images[0], test_labels[0]

## === cell 36
def plot_pred(prediction_probabilities, labels, images, n=1):
    """
    View the prediction, ground truth and image for sample n
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
    
    plt.title("{} {:2.0f}% {}".format(pred_label, np.max(pred_prob) * 100, true_label), color=color)

plot_pred(prediction_probabilities=tf.nn.softmax(preds, axis=1),
          labels=test_labels,
          images=test_images,
          n=10)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/372451764.py in <cell line: 0>()
     18     plt.title("{} {:2.0f}% {}".format(pred_label, np.max(pred_prob) * 100, true_label), color=color)
     19 
---> 20 plot_pred(prediction_probabilities=tf.nn.softmax(preds, axis=1),
     21           labels=test_labels,
     22           images=test_images,

NameError: name 'preds' is not defined

## === cell 37
def plot_pred_conf(prediction_probabilities, labels, n=1):
    """
    Plus the top 10 highest prediction confidences along with the truth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities[n], labels[n]
    
    pred_label = get_pred_label(pred_prob)
    
    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    
    top_10_plot = plt.barh(np.arange(len(top_10_pred_labels)), top_10_pred_values, color="grey")
    
    plt.gca().invert_yaxis()
    
    plt.yticks(np.arange(len(top_10_pred_labels)), labels=top_10_pred_labels)
    
    if np.isin(true_label, top_10_pred_labels):
        top_10_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")  

prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
plot_pred_conf(prediction_probabilities=prediction_probabilities,
               labels=test_labels,
               n=42)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2054483855.py in <cell line: 0>()
     24         top_10_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
     25 
---> 26 prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
     27 plot_pred_conf(prediction_probabilities=prediction_probabilities,
     28                labels=test_labels,

NameError: name 'preds' is not defined

## === cell 38
i_multiplier = 30
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(10*num_cols, 5*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_pred(prediction_probabilities=prediction_probabilities,
              labels=test_labels,
              images=test_images,
              n=i+i_multiplier)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_pred_conf(prediction_probabilities=prediction_probabilities,
                   labels=test_labels,
                   n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1935703047.py in <cell line: 0>()
      6 for i in range(num_images):
      7     plt.subplot(num_rows, 2*num_cols, 2*i+1)
----> 8     plot_pred(prediction_probabilities=prediction_probabilities,
      9               labels=test_labels,
     10               images=test_images,

NameError: name 'prediction_probabilities' is not defined

## === cell 39
import seaborn as sns

from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(test_labels, [get_pred_label(pred) for pred in prediction_probabilities])

plt.figure(figsize=(30,15))

ax = sns.heatmap(cf_matrix, 
                 cmap="Reds", 
                 linewidths=1, 
                 xticklabels=lb.classes_, 
                 yticklabels=lb.classes_)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.xaxis.tick_top()
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3487249945.py in <cell line: 0>()
      3 from sklearn.metrics import confusion_matrix
      4 
----> 5 cf_matrix = confusion_matrix(test_labels, [get_pred_label(pred) for pred in prediction_probabilities])
      6 
      7 plt.figure(figsize=(30,15))

NameError: name 'prediction_probabilities' is not defined

## === cell 41
save_model(model, file_name="mobilenetv2-Adam", include_datetime=False);

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2903869107.py in <cell line: 0>()
----> 1 save_model(model, file_name="mobilenetv2-Adam", include_datetime=False);

NameError: name 'model' is not defined

## === cell 43
submitted_data_dir = KAGGLE_DATA_DIR / 'test'
submitted_img_names = [str(filename) for filename in submitted_data_dir.iterdir()]
submitted_img_names[:2]

## === cell 44
submitted_data = create_data_batches(submitted_img_names, test_data=True)
next(submitted_data.as_numpy_iterator())[0][0][0]

## === cell 45
next(submitted_data_dir.iterdir()).stem

## === cell 46
submit_preds = tf.nn.softmax(model.predict(submitted_data, verbose=1), axis=1).numpy()

out_df = pd.DataFrame(columns=["id"]+list(unique_breeds))
submit_ids = [entry.stem for entry in submitted_data_dir.iterdir()]
out_df.loc[:,"id"] = submit_ids
out_df[list(unique_breeds)] = submit_preds
out_df.head()

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2393863124.py in <cell line: 0>()
----> 1 submit_preds = tf.nn.softmax(model.predict(submitted_data, verbose=1), axis=1).numpy()
      2 
      3 out_df = pd.DataFrame(columns=["id"]+list(unique_breeds))
      4 submit_ids = [entry.stem for entry in submitted_data_dir.iterdir()]
      5 out_df.loc[:,"id"] = submit_ids

NameError: name 'model' is not defined

## === cell 47
out_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1946670284.py in <cell line: 0>()
----> 1 out_df.to_csv('submission.csv', index=False)

NameError: name 'out_df' is not defined
