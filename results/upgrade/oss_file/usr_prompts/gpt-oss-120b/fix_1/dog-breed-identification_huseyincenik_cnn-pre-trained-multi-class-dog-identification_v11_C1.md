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

0.64296

# 6. Current score

Not yielded

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
from tensorflow.keras.layers import Dense, Activation


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



import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")


pd.set_option("display.max_columns", None)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def get_num_files(path):
    '''
    Counts the number of files in a folder.
    '''
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


## === cell 2
import os, cv2, random, time, shutil, csv
train_dir = '/kaggle/input/dog-breed-identification/train'
test_dir = '/kaggle/input/dog-breed-identification/test'
data_size = get_num_files(train_dir)
test_size = get_num_files(test_dir)
print('Data samples size: ', data_size)
print('Test samples size: ', test_size)


## === cell 3
labels_dataframe = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
sample_df = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
labels_dataframe.head(5)


## === cell 4
sample_df.head(5)


## === cell 5
dog_breeds = sorted(list(set(labels_dataframe['breed'])))
n_classes = len(dog_breeds)
print(n_classes)
dog_breeds[:5]


## === cell 6
class_to_num = dict(zip(dog_breeds, range(n_classes)))


## === cell 7
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


## === cell 8
from tensorflow.keras.preprocessing.image import load_img
from tqdm import tqdm
from keras.utils import to_categorical

img_size = (224,224, 3)
X, y = images_to_array(train_dir, labels_dataframe, img_size)


## === cell 9
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


## === cell 10
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)


## === cell 11
from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)


## === cell 12
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
nasnet_features = get_features(NASNetLarge,
                               nasnet_preprocessor,
                               img_size, X)


## === cell 13
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   img_size, X)


## === cell 14
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
vgg16_preprocessor = preprocess_input
vgg16_features = get_features(VGG16,
                                   vgg16_preprocessor,
                                   img_size, X)


## === cell 15
from keras.applications.resnet50 import ResNet50, preprocess_input
resnet50_preprocessor = preprocess_input
resnet50_features = get_features(ResNet50,
                                   resnet50_preprocessor,
                                   img_size, X)


## === cell 16
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
mobilenet_v2_preprocessor = preprocess_input
mobilenet_v2_features = get_features(MobileNetV2,
                                   mobilenet_v2_preprocessor,
                                   img_size, X)


## === cell 17
from keras.applications.densenet import DenseNet121, preprocess_input
densenet_preprocessor = preprocess_input
densenet_features = get_features(DenseNet121,
                                   densenet_preprocessor,
                                   img_size, X)


## === cell 18
del X


## === cell 19
final_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features,], axis=-1)
print('Final feature maps shape', final_features.shape)


## === cell 20
import matplotlib.pyplot as plt

original_value_counts = pd.Series(y.argmax(axis=1)).value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(final_features, y, test_size=0.2, stratify=y, random_state=42)

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

bar1 = ax.barh(index, original_value_counts, bar_width, label='Ana Data')
bar2 = ax.barh(index, train_value_counts, bar_width, label='Train Set', left=original_value_counts)
bar3 = ax.barh(index, val_value_counts, bar_width, label='Validation Set', left=original_value_counts + train_value_counts)
bar4 = ax.barh(index, test_value_counts, bar_width, label='Test Set', left=original_value_counts + train_value_counts + val_value_counts)

ax.set_xlabel('Oranlar')
ax.set_title('Sınıf Dağılımı')
ax.set_yticks(index)
ax.set_yticklabels(original_value_counts.index)
ax.legend()

plt.show()


## === cell 21
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(np.argmax(y_train, axis=1))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

class_weights_dict = {class_num: weight for class_num, weight in zip(np.unique(y_train_encoded), class_weights)}

print("Class Weights Dictionary:")
print(class_weights_dict)


## === cell 23
from keras.callbacks import EarlyStopping
EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]


## === cell 24
model_1 = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model_1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_1 = model_1.fit(X_train, y_train,
            batch_size=128,
            epochs=60,
            validation_data=(X_val, y_val),
            callbacks=my_callback,
            class_weight = class_weights_dict)


## === cell 25
loss, accuracy = model_1.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


## === cell 26
history_1


## === cell 27
history_data = history_1.history

loss_df_1 = pd.DataFrame(history_data)
loss_df_1


## === cell 28
loss_df_1.plot()
plt.show()


## === cell 29
from sklearn.metrics import confusion_matrix, classification_report

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


## === cell 31
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

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


## === cell 32
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score

y_test_binary = label_binarize(y_test, classes=range(n_classes))
y_pred_binary = label_binarize(y_pred, classes=range(n_classes))

model1_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')

model1_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')

model1_AP_micro = average_precision_score(label_binarize(y_test, classes=range(n_classes)),
                                          label_binarize(y_pred, classes=range(n_classes)),
                                          average='weighted')


print(f'Weighted-Averaged Precision: {model1_precision:.2f}')
print(f'Weighted-Averaged Recall: {model1_recall:.2f}')
print(f'Weighted-Averaged AP: {model1_AP_micro:.2f}')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4128108165.py in <cell line: 0>()
      4 # Sınıfları ikili formata dönüştür
      5 y_test_binary = label_binarize(y_test, classes=range(n_classes))
----> 6 y_pred_binary = label_binarize(y_pred, classes=range(n_classes))
      7 
      8 # Micro-averaging için precision_score kullanımı

NameError: name 'y_pred' is not defined

## === cell 38
final_features_1 = np.concatenate([vgg16_features,
                                 resnet50_features,
                                 mobilenet_v2_features,
                                 inc_resnet_features,], axis=-1)
print('Final feature maps shape', final_features_1.shape)


## === cell 47
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


## === cell 48
inception_features = get_features(InceptionV3, inception_preprocessor, img_size, test_data)
xception_features = get_features(Xception, xception_preprocessor, img_size, test_data)
nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, test_data)
inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, test_data)

test_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features],axis=-1)
print('Final feature maps shape', test_features.shape)


## === cell 49
del test_data


## === cell 50
y_pred = dnn.predict(test_features, batch_size=128)


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/453180758.py in <cell line: 0>()
      1 #Predict test labels given test data features.
----> 2 y_pred = dnn.predict(test_features, batch_size=128)

NameError: name 'dnn' is not defined

## === cell 51
for b in dog_breeds:
    sample_df[b] = y_pred[:,class_to_num[b]]
sample_df.to_csv('submission.csv', index=None)


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3857242079.py in <cell line: 0>()
      1 #Create submission file
      2 for b in dog_breeds:
----> 3     sample_df[b] = y_pred[:,class_to_num[b]]
      4 sample_df.to_csv('submission.csv', index=None)

NameError: name 'y_pred' is not defined

## === cell 55
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())


## === cell 56
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10));


## === cell 57
from IPython.display import display, Image
Image("/kaggle/input/dog-breed-identification/train/00693b8bc2470375cc744a6391d397ec.jpg")


## === cell 58
train_path = "../input/dog-breed-identification/train/"


## === cell 59
filenames = [train_path + fname + ".jpg" for fname in labels_csv["id"]]

filenames[:10]


## === cell 60
import os
if len(os.listdir(train_path)) == len(filenames):
  print("Filenames match actual amount of files!")
else:
  print("Filenames do not match actual amount of files, check the target directory.")


## === cell 61
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


## === cell 62
import numpy as np
labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array
labels[:20]


## === cell 63
if len(labels) == len(filenames):
  print("Number of labels matches number of filenames!")
else:
  print("Number of labels does not match number of filenames, check data directories.")


## === cell 64
unique_breeds = np.unique(labels)
len(unique_breeds)


## === cell 65
boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:2]


## === cell 66
print(labels[1]) # original label
print(np.where(unique_breeds == labels[1])[0][0]) # index where label occurs
print(boolean_labels[1].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs


## === cell 67
X = filenames
y = boolean_labels

print(f"Number of training images: {len(X)}")
print(f"Number of labels: {len(y)}")


## === cell 68
import pandas as pd

X = filenames
y = [np.where(label)[0][0] for label in boolean_labels]

train_df = pd.DataFrame({'image': X, 'label': y})

train_df.sample(10)


## === cell 69
list(train_df.iloc[1])


## === cell 76
nrow = 5
ncol = 4  # 20 resim olduğu için sütun sayısını 4 yapabilirsiniz.
fig1 = plt.figure(figsize=(20, 15))
fig1.suptitle('After Resizing', size=32)

for i in range(min(20, len(resized_image_list))):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(resized_image_list[i])
    plt.title('class = {x}, Dog is {y}'.format(x=train_df["label"].iloc[i], y=labels[i]))
    plt.axis('Off')
    plt.grid(False)
plt.show()


## --- ERROR in cell 76, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/582337322.py in <cell line: 0>()
      5 
      6 # İlk 20 resmi seç
----> 7 for i in range(min(20, len(resized_image_list))):
      8     plt.subplot(nrow, ncol, i + 1)
      9     plt.imshow(resized_image_list[i])

NameError: name 'resized_image_list' is not defined

## === cell 77
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.5)
], name='data_augmentation')


## === cell 78
augmented_images = data_augmentation(resized_image_list)


## --- ERROR in cell 78, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4194488576.py in <cell line: 0>()
----> 1 augmented_images = data_augmentation(resized_image_list)

NameError: name 'resized_image_list' is not defined

## === cell 82
nrow = 4
ncol = 5

augmented_indices = range(min(20, len(resized_image_list)))

fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('After Augmentation', size=32)

for i, idx in enumerate(augmented_indices):
    augmented_image = data_augmentation(tf.expand_dims(resized_image_list[idx], 0), training=True)
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(augmented_image[0].numpy())
    plt.title('class = {x}, Dog is {y}'.format(x=train_df["label"].iloc[idx], y=labels[idx]))
    plt.axis('Off')
    plt.grid(False)

plt.show()


## --- ERROR in cell 82, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/683773643.py in <cell line: 0>()
      3 
      4 # İlk 20 augmentasyonlu resmi seç
----> 5 augmented_indices = range(min(20, len(resized_image_list)))
      6 
      7 fig2 = plt.figure(figsize=(20, 15))

NameError: name 'resized_image_list' is not defined

## === cell 83
class_values = train_df["label"]
filtered_values = class_values[class_values < 0]

if not filtered_values.empty:
    print("There are values in the series less than 0.")
else:
    print("There are no values in the series less than 0.")
class_values.value_counts()


## === cell 84
augmented_images_tf = tf.convert_to_tensor(augmented_images)
selected_labels_tf = tf.convert_to_tensor(train_df['label'])

augmented_images_np = augmented_images_tf.numpy()
selected_labels_np = selected_labels_tf.numpy()

X_train, X_test, y_train, y_test = train_test_split(
    augmented_images_np, 
    selected_labels_np,
    test_size=0.3,
    stratify = selected_labels_np,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train,
    test_size=0.1,  # You can adjust the validation split as needed
    stratify = y_train,
    random_state=42
)

print("Training Set Length:", len(X_train))
print("Test Set Length:", len(X_test))                                  
print("Validation Set Length:", len(X_val))


## --- ERROR in cell 84, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1089066042.py in <cell line: 0>()
      1 # Assuming you have NumPy arrays for augmented_images and selected_labels
      2 # Convert NumPy arrays to TensorFlow tensors
----> 3 augmented_images_tf = tf.convert_to_tensor(augmented_images)
      4 selected_labels_tf = tf.convert_to_tensor(train_df['label'])
      5 

NameError: name 'augmented_images' is not defined

## === cell 86
def con_matrix(X_train, X_test, y_train, y_test, model, model_name='model'):

  y_pred = (model.predict(X_test) > 0.5).astype('int64')
  print(f'{model_name} - Test Confusion Matrix\n' + '*' * 50)
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

  print()

  y_train_pred = (model.predict(X_train) > 0.5).astype('int64')
  print(f'{model_name} - Train Confusion Matrix\n' + '*' * 50)
  print(confusion_matrix(y_train, y_train_pred))
  print(classification_report(y_train, y_train_pred))


## === cell 87
X_train.shape


## === cell 88
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization
)


## === cell 89
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3), padding = 'same'))

model.add(Conv2D(64, (3, 3), activation="relu", padding = 'same'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu", padding = 'same'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(140, activation="relu"))

model.add(Dense(200, activation="relu"))

model.add(Dense(120, activation="softmax"))

model.compile(optimizer="adam", 
              loss='sparse_categorical_crossentropy',
              metrics=['Recall'])




## === cell 90
model.summary()


## === cell 91
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size = 64)


## --- ERROR in cell 91, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2648989548.py in <cell line: 0>()
----> 1 model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size = 64)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: Exception encountered when calling Sequential.call().

Invalid input shape for input Tensor("data:0", shape=(None, 9664), dtype=float32). Expected shape (None, 64, 64, 3), but input has incompatible shape (None, 9664)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(None, 9664), dtype=float32)
  • training=True
  • mask=None

## === cell 92
pd.DataFrame(model.history.history).plot()
plt.show()


## --- ERROR in cell 92, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1484249843.py in <cell line: 0>()
----> 1 pd.DataFrame(model.history.history).plot()
      2 plt.show()

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py in __call__(self, *args, **kwargs)
   1028                     data.columns = label_name
   1029 
-> 1030         return plot_backend.plot(data, kind=kind, **kwargs)
   1031 
   1032     __call__.__doc__ = __doc__

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/__init__.py in plot(data, kind, **kwargs)
     69             kwargs["ax"] = getattr(ax, "left_ax", ax)
     70     plot_obj = PLOT_CLASSES[kind](data, **kwargs)
---> 71     plot_obj.generate()
     72     plot_obj.draw()
     73     return plot_obj.result

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py in generate(self)
    497     @final
    498     def generate(self) -> None:
--> 499         self._compute_plot_data()
    500         fig = self.fig
    501         self._make_plot(fig)

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py in _compute_plot_data(self)
    696         # no non-numeric frames or series allowed
    697         if is_empty:
--> 698             raise TypeError("no numeric data to plot")
    699 
    700         self.data = numeric_data.apply(type(self)._convert_to_ndarray)

TypeError: no numeric data to plot

## === cell 93
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


## --- ERROR in cell 93, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2576262203.py in <cell line: 0>()
----> 1 loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
      2 print("loss: ", loss)
      3 print("accuracy: ", accuracy)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: Exception encountered when calling Sequential.call().

Invalid input shape for input Tensor("data:0", shape=(None, 9664), dtype=float32). Expected shape (None, 64, 64, 3), but input has incompatible shape (None, 9664)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(None, 9664), dtype=float32)
  • training=False
  • mask=None

## === cell 94
pred_prob = model.predict(X_test)
y_pred = np.argmax(pred_prob, axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


## --- ERROR in cell 94, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3146964209.py in <cell line: 0>()
----> 1 pred_prob = model.predict(X_test)
      2 y_pred = np.argmax(pred_prob, axis=1)
      3 print(confusion_matrix(y_test, y_pred))
      4 print(classification_report(y_test, y_pred))

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: Exception encountered when calling Sequential.call().

Invalid input shape for input Tensor("data:0", shape=(32, 9664), dtype=float32). Expected shape (None, 64, 64, 3), but input has incompatible shape (32, 9664)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(32, 9664), dtype=float32)
  • training=False
  • mask=None

## === cell 95
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score

n_classes = len(set(y_test))

y_test_binary = label_binarize(y_test, classes=range(n_classes))
y_pred_binary = label_binarize(y_pred, classes=range(n_classes))

model1_AP_micro = average_precision_score(y_test_binary, y_pred_binary, average='weighted')

model1_precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
model1_recall = recall_score(y_test_binary, y_pred_binary, average='weighted')

print(f'Weighted-Averaged AP: {model1_AP_micro:.2f}')
print(f'Weighted-Averaged Precision: {model1_precision:.2f}')
print(f'Weighted-Averaged Recall: {model1_recall:.2f}')


## --- ERROR in cell 95, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/407721254.py in <cell line: 0>()
      4 
      5 # Sınıf sayısını belirle
----> 6 n_classes = len(set(y_test))
      7 
      8 # Sınıfları ikili formata dönüştür

TypeError: unhashable type: 'numpy.ndarray'

## === cell 96
n_classes
