# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.8

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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.28106

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import numpy as np
import pandas as pd
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output


from keras import backend as K
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Lambda,
    UpSampling2D,
    Concatenate,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.initializers import he_normal

from tf_keras.preprocessing.image import ImageDataGenerator

import shutil


## === cell 1
!ls ../input/dog-breed-identification/train/ -U | head -5 


## === cell 2
labels = pd.read_csv('../input/dog-breed-identification/labels.csv')
labels.head(5)


## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
classes_num


## === cell 4
train_dir = '../input/dog-breed-identification/train'  # the images directory
images_names = os.listdir(train_dir)  # names of the files in the directory
images_num = len(images_names)
print(f'Number of images: {images_num}')  # Number of training images


## === cell 5
new_train_dir = '/root/new_train/'  # parent directoiry of the training set
new_test_dir = '/root/new_test/'  # parent directory of the validation set
new_valid_dir = '/root/new_valid/'  # parent directory of the test set
!mkdir $new_train_dir
!mkdir $new_test_dir
!mkdir $new_valid_dir


## === cell 6
for sub_dir in classes:
    os.mkdir(new_train_dir+sub_dir)
    os.mkdir(new_test_dir+sub_dir)
    os.mkdir(new_valid_dir+sub_dir)
!ls $new_train_dir


## === cell 7
labels_jpg = labels.copy(deep=True)
labels_jpg['id'] += '.jpg'  # add .jpg to each image id to get its filename

grouped_ids = labels_jpg.groupby('breed')['id'].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]])


## === cell 8
test_split = 0.1
valid_split = 0.2


## === cell 9
train_size = 0
valid_size = 0
test_size = 0

for breed_idx, (breed, breed_images) in enumerate(grouped_ids.items()):
    for img in breed_images:
        rnd_prob = np.random.rand()  # give the current image a random number in the range [0, 1]
        if rnd_prob <= test_split: 
            shutil.copy(train_dir+'/'+img, new_test_dir+'/'+breed) 
            test_size += 1
            
        elif rnd_prob <= (test_split + valid_split):
            shutil.copy(train_dir+'/'+img, new_valid_dir+'/'+breed)
            valid_size += 1
            
        else:
            shutil.copy(train_dir+'/'+img, new_train_dir+'/'+breed)
            train_size += 1
            
    clear_output(wait=True)
    print(f'Organized {breed_idx+1} out of {classes_num} breeds: {breed}')


## === cell 10
print(train_size, valid_size, test_size)


## === cell 11
test_breed = classes[0]
!ls $new_train_dir/$test_breed | head -5


## === cell 12
width, height, channels = 512, 512, 3


## === cell 13
images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = np.random.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(train_dir + '/' + img_filename)  # loads the images channels in (blue, green, red) order
    images_samples[i] = cv2.resize(src=img_bgr[:, :, [2, 1, 0]], dsize=(width, height)) / 255  # store the random image
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)  # store the random images' label


## === cell 14
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Class: {label}', size=15);


## === cell 15
norm_factor = 1 / 255

transform_params = {
    'featurewise_center': False,
    'featurewise_std_normalization': False,
    'samplewise_center': False,
    'samplewise_std_normalization': False,
    'rotation_range': 30, 
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'horizontal_flip': True,
    'rescale': norm_factor
}

img_gen = ImageDataGenerator(**transform_params) 


## === cell 16
img_feed = ImageDataGenerator(rescale=1/255)


## === cell 17
fig, axs = plt.subplots(2, 4, figsize=(20,10))  # let's see 4 augmentation examples
fig.suptitle('Augmentation Results', size=32)

for axs_col, img in enumerate(images_samples):
    viz_transoform_params = {  # defined each iteration to get new augmentation values each time
        'theta': np.random.randint(-transform_params['rotation_range'], transform_params['rotation_range']),
        'tx': np.random.uniform(0, transform_params['width_shift_range']),
        'ty': np.random.uniform(0, transform_params['height_shift_range']),
        'flip_horizontal': np.random.choice([True, False], p=[0.5, 0.5])
    }

    aug_img = img_gen.apply_transform(img, viz_transoform_params)  # the same image after augmentation
    
    axs[0, axs_col].imshow(img);
    axs[0, axs_col].axis('off')
    axs[0, axs_col].set_title('Original Image', size=15)
    
    axs[1, axs_col].imshow(aug_img);
    axs[1, axs_col].axis('off')
    axs[1, axs_col].set_title('Augmented Image', size=15)


## === cell 18
class Plotter(Callback):
    def plot(self):  # Updates the graph
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.plot(self.epochs, self.losses, label='train_loss')
        ax1.plot(self.epochs, self.val_losses, label='val_loss')
        
        ax2.plot(self.epochs, self.acc, label='train_acc')
        ax2.plot(self.epochs, self.val_acc, label='val_acc')
    
        ax1.set_title(f'Loss vs Epochs')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        
        ax2.set_title(f'Accuracy vs Epochs')
        ax2.set_xlabel("Epoches")
        ax2.set_ylabel("Accuracy")
        
        ax1.legend()
        ax2.legend()
        plt.show()
        
        print(f'Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}')
        print(f'Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}')
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.batch_no = []
        self.acc = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.epochs.append(epoch)
        self.plot()  # update the graph
        
    def on_train_end(self, logs={}):
        self.plot()
        
    def load_plot_data(self, data):
        self.losses, self.val_losses, self.epochs, self.batch_no, self.acc, self.val_acc = data
    
    def get_plot_data(self):
        return [self.losses, self.val_losses, self.epochs, self.batch_no, self.acc, self.val_acc]
               
plotter = Plotter()


## === cell 19
plateau_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.01,
                              patience=1, min_lr=1e-20)


## === cell 20
e_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)


## === cell 21
callbacks = [plotter, plateau_reduce, e_stop]


## === cell 22
def dense_block(x, neurons, layer_no):
    x = Dense(neurons, kernel_initializer=he_normal(layer_no), name=f'topDense{layer_no}')(x)
    x = Activation('relu', name=f'Relu{layer_no}')(x)
    x = BatchNormalization(name=f'BatchNorm{layer_no}')(x)
    x = Dropout(0.5, name=f'Dropout{layer_no}')(x)
    return x


## === cell 23
def create_model(shape):
    input_layer = Input(shape, name='input_layer')  # input layer with given shape
    
    incep_res = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_layer)
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding='same')(incep_res.output)
    flat1 = Flatten(name='Flatten1')(pool)
    flat1_bn = BatchNormalization(name='BatchNormFlat')(flat1)
 
    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)
    
    dens_final = Dense(classes_num, name='Dense4')(dens3)
    output_layer = Activation('softmax', name='Softmax')(dens_final)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model


## === cell 24
height, width, channels_num = 512, 512, 3
learning_rate = 0.004
epochs = 15
batch_size = 32  # if increased you may run out of gpu memory - reduce image size if to increase the batch size


## === cell 25
model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.summary()


## === cell 26
train_gen = img_gen.flow_from_directory(directory=new_train_dir, target_size=(height, width), color_mode='rgb', classes=list(classes), 
                                        class_mode='categorical', batch_size=batch_size, shuffle=True, interpolation='nearest')

valid_gen = img_feed.flow_from_directory(directory=new_valid_dir, target_size=(height, width), color_mode='rgb', classes=list(classes), 
                                        class_mode='categorical', batch_size=batch_size, shuffle=True, interpolation='nearest')




## === cell 27
!wget "https://drive.google.com/uc?id=1-3Y-DB5uhOaY69pvVl5rmfirw9aKCYYB&export=download" -O '/root/training_curves.npy'


## === cell 28
plot_data = np.load('/root/training_curves.npy', allow_pickle=True)
plotter.load_plot_data(plot_data)  # load the data into the plotter
plotter.plot()


## === cell 30
!wget "https://drive.google.com/uc?id=1Omp4wFOWc2WslToduWAgBkyRv9poQVKb&export=download" -O '/root/decoder_weights.npy'


## === cell 31
decoder_weights = np.load('/root/decoder_weights.npy', allow_pickle=True).item()
for layer_name, layer_weights in decoder_weights.items():
    model.get_layer(layer_name).set_weights(layer_weights)  # set each layer of the decoder with its weights


## === cell 32
decoder_weights.keys()


## === cell 33
test_gen = ImageDataGenerator(rescale=1/255)
test_flow = test_gen.flow_from_directory(new_test_dir,
        target_size=(512, 512),
        batch_size=1,
        shuffle=False)


## === cell 34
import tensorflow as tf

output_signature = (
    tf.TensorSpec(shape=(None, height, width, channels_num), dtype=tf.float32),
    tf.TensorSpec(shape=(None, classes_num), dtype=tf.float32),
)

test_ds = tf.data.Dataset.from_generator(
    lambda: test_flow, output_signature=output_signature
)

metrics = model.evaluate(test_ds, steps=test_size)


## === cell 35
m_names = model.metrics_names
print(f'{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}')


## === cell 36
one_hot_map = train_gen.class_indices
one_hot_map


## === cell 37
input_dir = '../input/dog-breed-identification'


## === cell 38
test_names = os.listdir(input_dir+'/test')  # names of the files in the directory
test_names.sort()
test_size = len(test_names)
test_size  # number of test images to predict their labels


## === cell 39
test_flow = test_gen.flow_from_directory(input_dir,
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        classes=['test'])  # added test folder as class because keras' flow needs subdirectories hierarchy


## === cell 47
!wget 'https://drive.google.com/uc?id=18IQ9WVf1KcVTgwTgGZXKzx61W0ISayI8&export=download' -O 'submission_colab.csv'
