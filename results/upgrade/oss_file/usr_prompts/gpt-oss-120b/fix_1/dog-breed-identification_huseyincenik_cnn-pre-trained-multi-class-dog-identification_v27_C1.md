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

3.12

# 3. Installed packages

colorama==0.4.6
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scikit-plot==0.3.7
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
termcolor==3.1.0
tf_keras==2.18.0
tqdm==4.67.1

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

0.35444

# 6. Current score

0.95164

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, cv2, random, time, shutil, csv


plt.rcParams["figure.figsize"] = (10, 6)

sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: "%.3f" % x)



import colorama
from colorama import Fore, Style  # makes strings colored
from termcolor import colored
from termcolor import cprint

from tensorflow import keras
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import regularizers
from tensorflow.keras.preprocessing.image import load_img
from tqdm import tqdm
from keras.utils import to_categorical
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization
)


from sklearn.model_selection import cross_val_score, cross_validate 
from sklearn.metrics import RocCurveDisplay,accuracy_score, f1_score, recall_score,\
                            precision_score, make_scorer,\
                            classification_report,confusion_matrix,\
                            ConfusionMatrixDisplay, average_precision_score,\
                            roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from scikitplot.metrics import plot_roc, precision_recall_curve,average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize




import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")


pd.set_option("display.max_columns", None)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())


## === cell 2
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10));


## === cell 3
from IPython.display import display, Image
Image("/kaggle/input/dog-breed-identification/train/00693b8bc2470375cc744a6391d397ec.jpg")


## === cell 4
train_path = "../input/dog-breed-identification/train/"


## === cell 5
filenames = [train_path + fname + ".jpg" for fname in labels_csv["id"]]

filenames[:10]


## === cell 6
import os
if len(os.listdir(train_path)) == len(filenames):
  print("Filenames match actual amount of files!")
else:
  print("Filenames do not match actual amount of files, check the target directory.")


## === cell 7
from PIL import Image
import random


random_images = random.sample(filenames, 9)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    
    img_path = random_images[i]
    img = Image.open(img_path)
    label = labels_csv[labels_csv["id"] == os.path.splitext(os.path.basename(img_path))[0]]["breed"].values[0]
    
    img = img.resize((100, 100))
    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()
plt.show()


## === cell 8
import numpy as np
labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array
labels[:20]


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
print(labels[1]) # original label
print(np.where(unique_breeds == labels[1])[0][0]) # index where label occurs
print(boolean_labels[1].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs


## === cell 13
X = filenames
y = boolean_labels

print(f"Number of training images: {len(X)}")
print(f"Number of labels: {len(y)}")


## === cell 14
import pandas as pd

X = filenames
y = [np.where(label)[0][0] for label in boolean_labels]

train_df = pd.DataFrame({'image': X, 'label': y})

train_df.sample(10)


## === cell 15
list(train_df.iloc[1])


## === cell 17






def cnn_resize(data_df, img_size=(224, 224, 3)):
    """
    Load and process images from a DataFrame for CNN.

    Args:
        data_df (pd.DataFrame): DataFrame with 'image' and 'label' columns.
        img_size (tuple): Target size for the images.

    Returns:
        np.ndarray: Processed image array.
        np.ndarray: One-hot encoded label array.
    """
    images_paths = data_df['image']
    labels = data_df['label']
    data_size = len(images_paths)

    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_size, 1], dtype=np.uint8)

    for i in tqdm(range(data_size)):
        image_path = images_paths.iloc[i]  # Full path to image
        img_pixels = load_img(image_path, target_size=img_size)
        X[i] = img_pixels

        label = labels.iloc[i]

        if label not in labels.unique():
            continue

        y[i] = labels.unique().tolist().index(label)

    y = to_categorical(y)
    
    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]

    print('Output Data Size: ', X.shape)
    print('Output Label Size: ', y.shape)

    return X, y

img_size = (128, 128, 3)
X, y = cnn_resize(train_df, img_size=img_size)


## === cell 18
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Available GPUs:", gpus)


## === cell 19
nrow = 5
ncol = 4  
fig1 = plt.figure(figsize=(20, 15))
fig1.suptitle('After Resizing', size=32)

for i in range(min(20, len(X))):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(X[i])
    plt.title('class = {x}, Dog is {y}'.format(x=train_df["label"].iloc[i], y=labels[i]))
    plt.axis('Off')
    plt.grid(False)
plt.show()


## === cell 26
class_values = train_df["label"]
filtered_values = class_values[class_values < 0]

if not filtered_values.empty:
    print("There are values in the series less than 0.")
else:
    print("There are no values in the series less than 0.")
class_values.value_counts()


## === cell 27





original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 28
from sklearn.utils.class_weight import compute_class_weight

class_weights_manual = len(y_train) / (len(np.unique(y_train)) * np.array([len(np.where(y_train[:, i])[0]) for i in range(y_train.shape[1])]))
class_weights_dict = {class_num: weight for class_num, weight in enumerate(class_weights_manual)}

print("Class Weights Dictionary (Manually Calculated):")
print(class_weights_dict)


## === cell 29
X_train.shape


## === cell 30
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))




model.add(Flatten())

model.add(Dense(140, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(120, activation="softmax"))

model.compile(optimizer="adam", 
               loss='categorical_crossentropy',
               metrics=['Recall'])


## === cell 31
model.summary()


## === cell 32
EarlyStop_callback = EarlyStopping(monitor='val_loss',verbose = 1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 33
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=400, 
          batch_size = 128, callbacks=my_callback, class_weight = class_weights_dict)


## === cell 34
pd.DataFrame(model.history.history).plot()
plt.show()


## === cell 35
loss, recall = model.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("recall: ", recall)


## === cell 36
y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model.predict(X_train)
test_pred_prob = model.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 37

n_classes = 120

y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model_precision:.2f}')
print(f'Weighted-Averaged Recall: {model_recall:.2f}')
print(f'Weighted-Averaged AP: {model_AP:.2f}')


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2186587841.py in <cell line: 0>()
      6 # Assuming y_test_indices and y_test_pred are obtained as mentioned in your code
      7 # Convert to binary format
----> 8 y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
      9 y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))
     10 

NameError: name 'label_binarize' is not defined

## === cell 38
del X,y


## === cell 39
def get_num_files(path):
    '''
    Counts the number of files in a folder.
    '''
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


## === cell 40
train_dir = '/kaggle/input/dog-breed-identification/train'
test_dir = '/kaggle/input/dog-breed-identification/test'
data_size = get_num_files(train_dir)
test_size = get_num_files(test_dir)
print('Data samples size: ', data_size)
print('Test samples size: ', test_size)


## === cell 41
labels_dataframe = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
sample_df = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
labels_dataframe.head(5)


## === cell 42
sample_df.head(5)


## === cell 43
dog_breeds = sorted(list(set(labels_dataframe['breed'])))
n_classes = len(dog_breeds)
print(n_classes)
dog_breeds[:5]


## === cell 44
class_to_num = dict(zip(dog_breeds, range(n_classes)))


## === cell 45
def images_to_array(data_dir, labels_dataframe, img_size = (224,224,3)):
    '''
    1- Read image samples from certain directory.
    2- Risize it, then stack them into one big numpy array.
    3- Read sample's label form the labels dataframe.
    4- One hot encode labels array.
    5- Shuffle Data and label arrays.
    '''
    images_names = labels_dataframe['id']
    images_labels = labels_dataframe['breed']
    data_size = len(images_names)
    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_size,1], dtype=np.uint8)
    for i in tqdm(range(data_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name+'.jpg')
        img_pixels = load_img(img_dir, target_size=img_size)
        X[i] = img_pixels
        
        image_breed = images_labels[i]
        y[i] = class_to_num[image_breed]
    
    y = to_categorical(y)
    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]
    print('Ouptut Data Size: ', X.shape)
    print('Ouptut Label Size: ', y.shape)
    return X, y


## === cell 46

img_size = (300,300, 3)
X, y = images_to_array(train_dir, labels_dataframe, img_size)


## === cell 47
def get_features(model_name, data_preprocessor, input_size, data):
    '''
    1- Create a feature extractor to extract features from the data.
    2- Returns the extracted features and the feature extractor.
    '''
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


## === cell 48
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)


## === cell 49
from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)


## === cell 50
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
nasnet_features = get_features(NASNetLarge,
                               nasnet_preprocessor,
                               img_size, X)


## === cell 51
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   img_size, X)


## === cell 52
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
vgg16_preprocessor = preprocess_input
vgg16_features = get_features(VGG16,
                                   vgg16_preprocessor,
                                   img_size, X)


## === cell 53
from keras.applications.resnet50 import ResNet50, preprocess_input

resnet50_preprocessor = preprocess_input
resnet50_features = get_features(ResNet50,
                                   resnet50_preprocessor,
                                   img_size, X)


## === cell 54
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
mobilenet_v2_preprocessor = preprocess_input
mobilenet_v2_features = get_features(MobileNetV2,
                                   mobilenet_v2_preprocessor,
                                   img_size, X)


## === cell 55
from keras.applications.densenet import DenseNet121, preprocess_input
densenet_preprocessor = preprocess_input
densenet_features = get_features(DenseNet121,
                                   densenet_preprocessor,
                                   img_size, X)


## === cell 56
del X


## === cell 57
final_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features,], axis=-1)
print('Final feature maps shape', final_features.shape)


## === cell 58
original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features, y, test_size=0.3, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 59
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3209871214.py in <cell line: 0>()
----> 1 label_encoder = LabelEncoder()
      2 y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))
      3 
      4 class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
      5 

NameError: name 'LabelEncoder' is not defined

## === cell 60
batch_size = 64
epochs = 400


## === cell 61
EarlyStop_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 62
model_1 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Dropout(0.5),
    Dense(n_classes, activation='softmax', #kernel_regularizer=regularizers.l2(0.01)
         )])

model_1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['Recall'])

history_1 = model_1.fit( #final_features, y,
            X_train, y_train,
            batch_size= batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict
                       )


## === cell 63
loss, recall = model_1.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("recall: ", recall)


## === cell 64
history_1


## === cell 65
history_data = history_1.history

loss_df_1 = pd.DataFrame(history_data)
loss_df_1


## === cell 66
loss_df_1.plot()
plt.show()


## === cell 67
y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model_1.predict(X_train)
test_pred_prob = model_1.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 68
y_pred_proba = model_1.predict(X_test)

average_precisions = []
roc_aucs = []

for i in range(n_classes):  # n_classes is the number of classes in your problem
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
    roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

plt.figure(figsize=(20, 20))

for i in range(n_classes):
    plt.plot(recall, precision, lw=2, label=f'Class {i + 1} (AP = {average_precisions[i]:.2f}, AUC = {roc_aucs[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')

legend_data = {'Class': [f'Class {i + 1}' for i in range(n_classes)],
               'Average Precision': average_precisions,
               'AUC': roc_aucs}
legend_df = pd.DataFrame(legend_data)
plt.show()
legend_df


## --- ERROR in cell 68, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2755346458.py in <cell line: 0>()
      7 
      8 for i in range(n_classes):  # n_classes is the number of classes in your problem
----> 9     precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
     10     average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
     11     roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

NameError: name 'precision_recall_curve' is not defined

## === cell 69

y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model1_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model1_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model1_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model1_precision:.2f}')
print(f'Weighted-Averaged Recall: {model1_recall:.2f}')
print(f'Weighted-Averaged AP: {model1_AP:.2f}')


## --- ERROR in cell 69, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2168717551.py in <cell line: 0>()
      4 # Assuming y_test_indices and y_test_pred are obtained as mentioned in your code
      5 # Convert to binary format
----> 6 y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
      7 y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))
      8 

NameError: name 'label_binarize' is not defined

## === cell 70
final_features_1 = np.concatenate([vgg16_features,
                                 resnet50_features,
                                 mobilenet_v2_features,
                                 inc_resnet_features,], axis=-1)
print('Final feature maps shape', final_features_1.shape)


## === cell 71
original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features_1, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 72

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1416109477.py in <cell line: 0>()
      2 # from sklearn.preprocessing import LabelEncoder
      3 
----> 4 label_encoder = LabelEncoder()
      5 y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))
      6 

NameError: name 'LabelEncoder' is not defined

## === cell 73
from keras.callbacks import EarlyStopping
EarlyStop_callback = EarlyStopping(monitor='val_loss',verbose=1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 74
model_2 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model_2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['Recall'])

history_2 = model_2.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict)


## === cell 75
loss, recall = model_2.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("recall: ", recall)


## === cell 76
history_2


## === cell 77
history_data = history_2.history

loss_df_2 = pd.DataFrame(history_data)
loss_df_2


## === cell 78
loss_df_2.plot()
plt.show()


## === cell 79

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model_2.predict(X_train)
test_pred_prob = model_2.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 80
y_pred_proba = model_2.predict(X_test)

average_precisions = []
roc_aucs = []

for i in range(n_classes):  # n_classes is the number of classes in your problem
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
    roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

plt.figure(figsize=(20, 20))

for i in range(n_classes):
    plt.plot(recall, precision, lw=2, label=f'Class {i + 1} (AP = {average_precisions[i]:.2f}, AUC = {roc_aucs[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')

legend_data = {'Class': [f'Class {i + 1}' for i in range(n_classes)],
               'Average Precision': average_precisions,
               'AUC': roc_aucs}
legend_df = pd.DataFrame(legend_data)
plt.show()
legend_df


## --- ERROR in cell 80, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/440200550.py in <cell line: 0>()
      7 
      8 for i in range(n_classes):  # n_classes is the number of classes in your problem
----> 9     precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
     10     average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
     11     roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

NameError: name 'precision_recall_curve' is not defined

## === cell 81
y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model2_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model2_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model2_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model2_precision:.2f}')
print(f'Weighted-Averaged Recall: {model2_recall:.2f}')
print(f'Weighted-Averaged AP: {model2_AP:.2f}')


## --- ERROR in cell 81, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2536241075.py in <cell line: 0>()
      1 # Assuming y_test_indices and y_test_pred are obtained as mentioned in your code
      2 # Convert to binary format
----> 3 y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
      4 y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))
      5 

NameError: name 'label_binarize' is not defined

## === cell 82
final_features_3 = np.concatenate([vgg16_features,
                                 resnet50_features,
                                 mobilenet_v2_features,
                                 densenet_features], axis=-1)
print('Final feature maps shape', final_features_3.shape)


## === cell 83
original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features_3, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 84

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## --- ERROR in cell 84, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1416109477.py in <cell line: 0>()
      2 # from sklearn.preprocessing import LabelEncoder
      3 
----> 4 label_encoder = LabelEncoder()
      5 y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))
      6 

NameError: name 'LabelEncoder' is not defined

## === cell 85
EarlyStop_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 86
model_3 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model_3.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['Recall'])

history_3 = model_3.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict)


## === cell 87
loss, accuracy = model_3.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


## === cell 88
history_3


## === cell 89
history_data = history_3.history

loss_df_3 = pd.DataFrame(history_data)
loss_df_3


## === cell 90
loss_df_3.plot()
plt.show()


## === cell 91

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model_3.predict(X_train)
test_pred_prob = model_3.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 92
y_pred_proba = model_3.predict(X_test)

average_precisions = []
roc_aucs = []

for i in range(n_classes):  # n_classes is the number of classes in your problem
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
    roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

plt.figure(figsize=(20, 20))

for i in range(n_classes):
    plt.plot(recall, precision, lw=2, label=f'Class {i + 1} (AP = {average_precisions[i]:.2f}, AUC = {roc_aucs[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')

legend_data = {'Class': [f'Class {i + 1}' for i in range(n_classes)],
               'Average Precision': average_precisions,
               'AUC': roc_aucs}
legend_df = pd.DataFrame(legend_data)
plt.show()
legend_df


## --- ERROR in cell 92, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/281452722.py in <cell line: 0>()
      7 
      8 for i in range(n_classes):  # n_classes is the number of classes in your problem
----> 9     precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
     10     average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
     11     roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

NameError: name 'precision_recall_curve' is not defined

## === cell 93

y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model3_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model3_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model3_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model3_precision:.2f}')
print(f'Weighted-Averaged Recall: {model3_recall:.2f}')
print(f'Weighted-Averaged AP: {model3_AP:.2f}')


## --- ERROR in cell 93, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/938787203.py in <cell line: 0>()
      4 # Assuming y_test_indices and y_test_pred are obtained as mentioned in your code
      5 # Convert to binary format
----> 6 y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
      7 y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))
      8 

NameError: name 'label_binarize' is not defined

## === cell 94
final_features_4 = np.concatenate([inception_features,
                                 resnet50_features,
                                 nasnet_features,
                                 densenet_features], axis=-1)
print('Final feature maps shape', final_features_4.shape)


## === cell 95
original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features_4, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 96


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## --- ERROR in cell 96, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/163687754.py in <cell line: 0>()
      3 
      4 
----> 5 label_encoder = LabelEncoder()
      6 y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))
      7 

NameError: name 'LabelEncoder' is not defined

## === cell 97
EarlyStop_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 98
model_4 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model_4.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['Recall'])

history_4 = model_4.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict)


## === cell 99
loss, recall = model_4.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("recall: ", accuracy)


## === cell 100
history_4


## === cell 101
history_data = history_4.history

loss_df_4 = pd.DataFrame(history_data)
loss_df_4


## === cell 102
loss_df_4.plot()
plt.show()


## === cell 103

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model_4.predict(X_train)
test_pred_prob = model_4.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 104
y_pred_proba = model_4.predict(X_test)

average_precisions = []
roc_aucs = []

for i in range(n_classes):  # n_classes is the number of classes in your problem
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
    roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

plt.figure(figsize=(20, 20))

for i in range(n_classes):
    plt.plot(recall, precision, lw=2, label=f'Class {i + 1} (AP = {average_precisions[i]:.2f}, AUC = {roc_aucs[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')

legend_data = {'Class': [f'Class {i + 1}' for i in range(n_classes)],
               'Average Precision': average_precisions,
               'AUC': roc_aucs}
legend_df = pd.DataFrame(legend_data)
plt.show()
legend_df


## --- ERROR in cell 104, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1203561954.py in <cell line: 0>()
      7 
      8 for i in range(n_classes):  # n_classes is the number of classes in your problem
----> 9     precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
     10     average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
     11     roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

NameError: name 'precision_recall_curve' is not defined

## === cell 105

y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model4_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model4_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model4_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model4_precision:.2f}')
print(f'Weighted-Averaged Recall: {model4_recall:.2f}')
print(f'Weighted-Averaged AP: {model4_AP:.2f}')


## --- ERROR in cell 105, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3814107905.py in <cell line: 0>()
      4 # Assuming y_test_indices and y_test_pred are obtained as mentioned in your code
      5 # Convert to binary format
----> 6 y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
      7 y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))
      8 

NameError: name 'label_binarize' is not defined

## === cell 106
final_features_5 = np.concatenate([mobilenet_v2_features,
                                 resnet50_features,
                                 vgg16_features,
                                 densenet_features], axis=-1)
print('Final feature maps shape', final_features_5.shape)


## === cell 107
original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features_5, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 108
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## --- ERROR in cell 108, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3209871214.py in <cell line: 0>()
----> 1 label_encoder = LabelEncoder()
      2 y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))
      3 
      4 class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
      5 

NameError: name 'LabelEncoder' is not defined

## === cell 109
EarlyStop_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 110
from keras.optimizers import Adam
from keras.regularizers import l2

model_5 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.00005)
model_5.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['Recall'])

history_5 = model_5.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict)


## === cell 111
loss, recall = model_5.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("recall: ", accuracy)


## === cell 112
history_5


## === cell 113
history_data = history_5.history

loss_df_5 = pd.DataFrame(history_data)
loss_df_5


## === cell 114
loss_df_5.plot()
plt.show()


## === cell 115

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model_5.predict(X_train)
test_pred_prob = model_5.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 116
y_pred_proba = model_5.predict(X_test)

average_precisions = []
roc_aucs = []

for i in range(n_classes):  # n_classes is the number of classes in your problem
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
    roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

plt.figure(figsize=(20, 20))

for i in range(n_classes):
    plt.plot(recall, precision, lw=2, label=f'Class {i + 1} (AP = {average_precisions[i]:.2f}, AUC = {roc_aucs[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')

legend_data = {'Class': [f'Class {i + 1}' for i in range(n_classes)],
               'Average Precision': average_precisions,
               'AUC': roc_aucs}
legend_df = pd.DataFrame(legend_data)
plt.show()
legend_df


## --- ERROR in cell 116, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1046988797.py in <cell line: 0>()
      7 
      8 for i in range(n_classes):  # n_classes is the number of classes in your problem
----> 9     precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
     10     average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
     11     roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

NameError: name 'precision_recall_curve' is not defined

## === cell 117
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize

y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model5_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model5_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model5_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model5_precision:.2f}')
print(f'Weighted-Averaged Recall: {model5_recall:.2f}')
print(f'Weighted-Averaged AP: {model5_AP:.2f}')


## === cell 118
final_features_6 = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                    ],axis=-1)
print('Final feature maps shape', final_features_6.shape)


## === cell 119
original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features_6, y, test_size=0.1, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, stratify=y_train, random_state=42)

y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_value_counts = pd.Series(y_train_indices).value_counts(normalize=True)
val_value_counts = pd.Series(y_val_indices).value_counts(normalize=True)
test_value_counts = pd.Series(y_test_indices).value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(20, 20))

bar_width = 0.2
index = np.arange(len(original_value_counts))

bar1 = ax.barh(index, original_value_counts, bar_width, label='Main Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Percentages')
ax.set_title('Class Distribution')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 120
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## --- ERROR in cell 120, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3209871214.py in <cell line: 0>()
----> 1 label_encoder = LabelEncoder()
      2 y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))
      3 
      4 class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
      5 

NameError: name 'LabelEncoder' is not defined

## === cell 121
batch_size = 64
epochs = 1000


## === cell 122
from keras.callbacks import EarlyStopping
EarlyStop_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 123
from keras import regularizers
model_6 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.001)
         )])

from keras.optimizers import Adam

custom_optimizer = Adam(learning_rate=0.005)

model_6.compile(optimizer=custom_optimizer,
              loss='categorical_crossentropy',
              metrics=['Recall'])


history_6 = model_6.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict
                       )


## === cell 124
from sklearn.metrics import log_loss
y_pred_proba = model_6.predict(X_val)

y_true = np.argmax(y_val, axis=1)

loss = log_loss(y_true, y_pred_proba)

print(f"Multi Class Log Loss: {loss}")


## === cell 125
loss, accuracy = model_6.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


## === cell 126
history_6


## === cell 127
history_data = history_6.history

loss_df_6 = pd.DataFrame(history_data)
loss_df_6


## === cell 128
loss_df_7.plot()
plt.show()


## --- ERROR in cell 128, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2704620365.py in <cell line: 0>()
----> 1 loss_df_7.plot()
      2 plt.show()

NameError: name 'loss_df_7' is not defined

## === cell 129

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

train_pred_prob = model_6.predict(X_train)
test_pred_prob = model_6.predict(X_test)

y_train_pred = np.argmax(train_pred_prob, axis=1)
y_test_pred = np.argmax(test_pred_prob, axis=1)

print("Training Dataset:")
print(confusion_matrix(y_train_indices, y_train_pred))
print(classification_report(y_train_indices, y_train_pred))

print("\nTest Dataset:")
print(confusion_matrix(y_test_indices, y_test_pred))
print(classification_report(y_test_indices, y_test_pred))


## === cell 130
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

y_pred_proba = model_6.predict(X_test)

average_precisions = []
roc_aucs = []

plt.figure(figsize=(20, 20))

for i in range(n_classes):  # n_classes is the number of classes in your problem
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision_score(y_test[:, i], y_pred_proba[:, i]))
    roc_aucs.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

    plt.plot(recall, precision, lw=2, label=f'Class {i + 1} (AP = {average_precisions[i]:.2f}, AUC = {roc_aucs[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.show()

legend_data = {'Class': [f'Class {i + 1}' for i in range(n_classes)],
               'Average Precision': average_precisions,
               'AUC': roc_aucs}
legend_df = pd.DataFrame(legend_data)
legend_df


## === cell 131

y_test_binary = label_binarize(y_test_indices, classes=range(n_classes))
y_pred_binary = label_binarize(y_test_pred, classes=range(n_classes))

model6_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model6_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
model6_AP = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged Precision: {model6_precision:.2f}')
print(f'Weighted-Averaged Recall: {model6_recall:.2f}')
print(f'Weighted-Averaged AP: {model6_AP:.2f}')


## === cell 132
model_names = ["Model"] + [f"Model {i}" for i in range(1, 7)]  # Change the range to include Model 6

compare = pd.DataFrame({
    "Model": model_names,
    "Precision": [model_precision, model1_precision, model2_precision, model3_precision, model4_precision, model5_precision, model6_precision], 
    "Recall": [model_recall, model1_recall, model2_recall, model3_recall, model4_recall, model5_recall, model6_recall],  
    "AP": [model_AP, model1_AP, model2_AP, model3_AP, model4_AP, model5_AP, model6_AP] 
})


## --- ERROR in cell 132, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1457992527.py in <cell line: 0>()
      3 compare = pd.DataFrame({
      4     "Model": model_names,
----> 5     "Precision": [model_precision, model1_precision, model2_precision, model3_precision, model4_precision, model5_precision, model6_precision],
      6     "Recall": [model_recall, model1_recall, model2_recall, model3_recall, model4_recall, model5_recall, model6_recall],
      7     "AP": [model_AP, model1_AP, model2_AP, model3_AP, model4_AP, model5_AP, model6_AP]

NameError: name 'model_precision' is not defined

## === cell 133
new_palette = "Reds" 

plt.figure(figsize=(14, 10))

plt.subplot(311)
compare_precision = compare.sort_values(by="Precision", ascending=False)
ax = sns.barplot(x="Precision", y="Model", data=compare_precision, palette=new_palette)
ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=10)
plt.title("Precision Comparison")

plt.subplot(312)
compare_recall = compare.sort_values(by="Recall", ascending=False)
ax = sns.barplot(x="Recall", y="Model", data=compare_recall, palette=new_palette)
ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=10)
plt.title("Recall Comparison")

plt.subplot(313)
compare_ap = compare.sort_values(by="AP", ascending=False)
ax = sns.barplot(x="AP", y="Model", data=compare_ap, palette=new_palette)
ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=10)
plt.title("Average Precision Comparison")

plt.tight_layout()
plt.show()


## --- ERROR in cell 133, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476266871.py in <cell line: 0>()
      4 
      5 plt.subplot(311)
----> 6 compare_precision = compare.sort_values(by="Precision", ascending=False)
      7 ax = sns.barplot(x="Precision", y="Model", data=compare_precision, palette=new_palette)
      8 ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=10)

NameError: name 'compare' is not defined

## === cell 134
fig, axs = plt.subplots(3, 3, figsize=(15, 8))

loss_df_1.plot(ax=axs[0, 0])
axs[0, 0].set_title('Model 0')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')

loss_df_1.plot(ax=axs[0, 1])
axs[0, 1].set_title('Model 1')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')

loss_df_2.plot(ax=axs[0, 2])
axs[0, 2].set_title('Model 2')
axs[0, 2].set_xlabel('Epochs')
axs[0, 2].set_ylabel('Loss')

loss_df_3.plot(ax=axs[1, 0])
axs[1, 0].set_title('Model 3')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Loss')

loss_df_4.plot(ax=axs[1, 1])
axs[1, 1].set_title('Model 4')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Loss')

loss_df_5.plot(ax=axs[1, 2])
axs[1, 2].set_title('Model 5')
axs[1, 2].set_xlabel('Epochs')
axs[1, 2].set_ylabel('Loss')

loss_df_6.plot(ax=axs[2, 0])
axs[2, 0].set_title('Model 6')
axs[2, 0].set_xlabel('Epochs')
axs[2, 0].set_ylabel('Loss')

plt.tight_layout()
plt.show()


## === cell 135
def images_to_array2(data_dir, labels_dataframe, img_size = (224,224,3)):
    '''
    Do same as images_to_array but omit some unnecessary steps for test data.
    '''
    images_names = labels_dataframe['id']
    data_size = len(images_names)
    X = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)
    
    for i in tqdm(range(data_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name+'.jpg')
        img_pixels = tf.keras.preprocessing.image.load_img(img_dir, target_size=img_size)
        X[i] = img_pixels
        
    print('Ouptut Data Size: ', X.shape)
    return X

test_data = images_to_array2(test_dir, sample_df, img_size)


## === cell 136
inception_features = get_features(InceptionV3, inception_preprocessor, img_size, test_data)
xception_features = get_features(Xception, xception_preprocessor, img_size, test_data)
nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, test_data)

test_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                ],axis=-1)
print('Final feature maps shape', test_features.shape)


## === cell 137
del test_data


## === cell 138
y_pred = model_6.predict(test_features, batch_size=128)


## === cell 139
for b in dog_breeds:
    sample_df[b] = y_pred[:,class_to_num[b]]
    
sample_df.to_csv('submission.csv', index=None)
