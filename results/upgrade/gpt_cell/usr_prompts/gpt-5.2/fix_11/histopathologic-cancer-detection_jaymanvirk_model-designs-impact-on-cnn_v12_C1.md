# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        if hasattr(_message_factory.MessageFactory, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

        else:
            def _GetPrototype(self, descriptor):
                return getattr(descriptor, "_concrete_class", None)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf

try:
    import tensorflow_io as tfio
except Exception:
    tfio = None

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from tensorflow.keras import layers, models

from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model


## === cell 1
input_dir = '/kaggle/input/histopathologic-cancer-detection'
list_l = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
list_l


## === cell 2
sample_csv_path = os.path.join(input_dir, "sample_submission.csv")
train_csv_path = os.path.join(input_dir, "train_labels.csv")
train_dir_path = os.path.join(input_dir, "train")
test_dir_path = os.path.join(input_dir, "test")

sample_data = pd.read_csv(sample_csv_path)
train_data = pd.read_csv(train_csv_path)

train_dir = train_dir_path + "/"
test_dir = test_dir_path + "/"

list_l = [sample_csv_path, train_csv_path, test_dir_path, train_dir_path]


## === cell 3
del list_l


## === cell 4
def print_short_summary(name, data):
    """
    Prints data head, shape and info.
    Args:
        name (str): name of dataset
        data (dataframe): dataset in a pd.DataFrame format
    """
    print(name)
    print('\n1. Data head:')
    print(data.head())
    print('\n2. Data shape: {}'.format(data.shape))
    print('\n3. Data info:')
    data.info()
    
def print_number_files(dirpath):
    print('{}: {} files'.format(dirpath, len(os.listdir(dirpath))))


## === cell 5
print_short_summary('Train data', train_data)


## === cell 6
print_short_summary('Sample data', sample_data)


## === cell 7
print_number_files(train_dir)


## === cell 8
print_number_files(test_dir)


## === cell 9
del print_short_summary, print_number_files


## === cell 10
plt.figure(figsize=(16, 9))
tmp = train_data['label'].value_counts()
sns.barplot(y=['No Cancer', 'Cancer'], x=tmp.values, orient='h')
plt.xlabel('Number of records')
plt.ylabel('Label')
plt.title('Number of records per label')
plt.show()


## === cell 11
del tmp


## === cell 12
def get_images_to_plot(file_names):
    """
    Returns list of images
    Args:
        file_names: list of filenames
    Returns:
        list of image objects
    """
    return [Image.open(f) for f in file_names]

def get_image_label(dirname, data, labels, n = 5):
    """
    Return dictionary with label-imagepath
    Args:
        dirname: name of the directory
        data: dataset of file names
        labels: list of labels
        n (opt): number of images per label
    Returns:
        dict_img: dictionary with label-imagepath pairs
    """
    dict_img = {}
    for l in labels:
        indexes = data['label'] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp['id'] + '.tif'
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp
        
    return dict_img


## === cell 13
img_path = train_dir + train_data['id'][0] + '.tif'
img = Image.open(img_path)
print('Original image size: {}'.format(img.size))


## === cell 14
data = get_image_label(train_dir,train_data, [0,1])


## === cell 15
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 9))

labels = ['No Cancer', 'Cancer']
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(data[row][col])
    axes[row, col].set_title(labels[row])
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()


## === cell 16
del get_images_to_plot, get_image_label, img_path, img
del data, fig, axes, labels, row, col


## === cell 17
no_cancer = train_data[train_data['label'] == 0]
cancer = train_data[train_data['label'] == 1]

no_cancer_downsampled = resample(no_cancer,
                              replace=False, 
                              n_samples=len(cancer),
                              random_state=0)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])

balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(drop=True)


## === cell 18
del no_cancer, cancer, no_cancer_downsampled


## === cell 19
image_paths = train_dir + balanced_train_data['id'] + '.tif'
image_paths = image_paths.values

labels = balanced_train_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(image_paths
                                                    , labels
                                                    , test_size = 0.25
                                                    , shuffle = True
                                                    , random_state = 0)


## === cell 20
del image_paths, labels


## === cell 21
def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images using TensorFlow I/O.
    Decode image with 4 channels RGBA.
    Resize image to 32x32px.
    Scale pixels from 0 to 1.
    Args:
        image_path: path to TIFF image
        label (optional): true label from train data
    Returns:
        (img, label): for train data
        img: for test data
    """
    img = tf.io.read_file(image_path)
    img = tfio.experimental.image.decode_tiff(img)
    img = tf.image.resize(img, [32, 32])
    img = tf.cast(img, tf.float32) / 255.0
    
    return img if label is None else (img, label)

def get_prefetched_data(data, batch_size):
    """
    Create a TensorFlow dataset from image paths and labels.
    Execution in parallel.
    Load, preprocess images and batch the data.
    Prefetch batches to improve training performance.
    Args:
        data (tuple): image paths and corresponding labels
        batch_size (int): number of samples per batch
    Returns:
        tf.data.Dataset: preprocessed and preloaded TensorFlow dataset for keras CNN
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


## === cell 22
if tfio is None:
    _use_tfio = False
else:
    try:
        _ = tfio.experimental.image.decode_tiff(tf.constant(b""))
        _use_tfio = True
    except Exception:
        _use_tfio = False

if not _use_tfio:

    def get_decoded_image(image_path, label=None):
        img_bytes = tf.io.read_file(image_path)

        def _decode_with_pil(x):
            import io
            import numpy as _np
            from PIL import Image as _Image

            im = _Image.open(io.BytesIO(x))
            im = im.convert("RGBA")
            return _np.array(im, dtype=_np.uint8)

        img = tf.numpy_function(_decode_with_pil, [img_bytes], tf.uint8)
        img.set_shape([None, None, 4])
        img = tf.image.resize(img, [32, 32])
        img = tf.cast(img, tf.float32) / 255.0
        return img if label is None else (img, label)


BATCH_SIZE = 128

train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE)


## === cell 23
del X_train, y_train, X_test, y_test


## === cell 24
def get_model_base():
    """
    Return base model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))

        , layers.Flatten()

        , layers.Dense(32, activation='relu')

        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 25
def get_model_base_deep():
    """
    Return deeper model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))
        , layers.Conv2D(32, (3, 3), activation='relu')

        , layers.Flatten()

        , layers.Dense(32, activation='relu')
        , layers.Dense(32, activation='relu')

        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 26
def get_model_base_wide():
    """
    Return wider model architecture
    """
    model_drop_bn = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 4))

        , layers.Flatten()
        
        , layers.Dense(64, activation='relu')
 
        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model_drop_bn


## === cell 27
def get_model_base_maxpool():
    """
    Return maxpool model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))
        , layers.MaxPooling2D((2, 2), strides = (2,2))

        , layers.Flatten()

        , layers.Dense(32, activation='relu')
        
        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 28
def get_model_base_dropout():
    """
    Return dropout model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))

        , layers.Flatten()

        , layers.Dense(32, activation='relu')
        
        , layers.Dropout(0.25)
        
        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 29
def get_compiled_model(func):
    """
    Create model to be trained with a multi-GPU strategy.
    Args:
        func: function to get model architecture
    Returns:
        compiled_model: tensorflow model that performs data parallelism
                            by copying all of the model's variables
                            to each processor
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()

        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print('No GPU available, falling back to CPU.')

    with strategy.scope():
        compiled_model = func()
        compiled_model.compile(optimizer = tf.keras.optimizers.Adam()
                              , loss = tf.keras.losses.BinaryCrossentropy()
                              , metrics = [tf.keras.metrics.AUC()])

    return compiled_model


## === cell 30
def plot_model_scores(scores, model_name):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, label='Train score')
    plt.plot(epochs, test_scores, label='Test score')
    plt.title('Train and test ROC AUC scores of the {}'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    
def get_model_results(model_name, model):
    """
    Return tuple of runtime, train and test scores.
    Compile, fit and save model along the way.
    Args:
        model: fitted model
    Returns:
        (runtime, (train_scores, test_scores) )
    """
    model = get_compiled_model(model)
    
    st = time.time()
    model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    runtime = time.time() - st
    
    model.save('{}.h5'.format(model_name))
    
    train_scores = model.history.history['auc']
    test_scores = model.history.history['val_auc']
    
    tf.keras.backend.clear_session()
    
    return (runtime, (train_scores, test_scores))


## === cell 31
runtime_base, scores_base = get_model_results('model_base',get_model_base)


## === cell 32
plot_model_scores(scores_base, 'base model')


## === cell 33
runtime_base_deep, scores_base_deep = get_model_results('model_base_deep'
                                                          ,get_model_base_deep)


## === cell 34
plot_model_scores(scores_base_deep, 'base + additional layers model')


## === cell 35
runtime_base_wide, scores_base_wide = get_model_results('model_base_wide'
                                                          ,get_model_base_wide)


## === cell 36
plot_model_scores(scores_base_wide, 'base + wider layers model')


## === cell 37
runtime_base_maxpool, scores_base_maxpool = get_model_results('model_base_maxpool'
                                                                ,get_model_base_maxpool)


## === cell 38
plot_model_scores(scores_base_maxpool, 'base + max pooling model')


## === cell 39
runtime_base_dropout, scores_base_dropout = get_model_results('model_base_dropout'
                                                                ,get_model_base_dropout)


## === cell 40
plot_model_scores(scores_base_dropout, 'base + dropout model')


## === cell 41
results = [('Base', runtime_base, scores_base)
          ,('Base + Add. layers', runtime_base_deep, scores_base_deep)
          ,('Base + Wider layers', runtime_base_wide, scores_base_wide)
          ,('Base + Max pooling', runtime_base_maxpool, scores_base_maxpool)
          ,('Base + Dropout', runtime_base_dropout, scores_base_dropout)]
table = []
for i in range(len(results)):
    tmp = {
            'model': results[i][0]
            , 'runtime': results[i][1]
            , 'train_roc_auc_score': results[i][2][0][-1]
            , 'test_roc_auc_score': results[i][2][1][-1]
        }
    table.append(tmp)


pd.DataFrame(table).sort_values(by = ['test_roc_auc_score'
                                      ,'runtime']
                                , ascending = [False, True])


## === cell 42
del results, tmp, table, train_dataset, test_dataset, train_data, train_dir, test_dir
del get_model_results, plot_model_scores, get_compiled_model


## === cell 43
model = load_model('model_base_maxpool.h5')


## === cell 44
submis_data = test_dir + sample_data['id'] + '.tif'
submis_data = submis_data.values

submis_dataset = get_prefetched_data((submis_data)
                                    , BATCH_SIZE)


## === cell 45
results = model.predict(submis_dataset)


## === cell 46
sample_data['label'] = np.ravel(np.round(results))


## === cell 47
sample_data


## === cell 48
sample_data.to_csv('submission.csv', index=False)


## === cell 49
del submis_data, submis_dataset, sample_data
del get_decoded_image, get_prefetched_data
