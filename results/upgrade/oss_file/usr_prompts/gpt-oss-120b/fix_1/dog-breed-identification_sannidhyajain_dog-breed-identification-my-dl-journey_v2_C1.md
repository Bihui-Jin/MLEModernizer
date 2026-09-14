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

0.99749

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install tensorflow
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
cpu()
gpu()

print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))


## === cell 1
import tensorflow as tf
import tensorflow_hub as hub
tf.__version__, hub.__version__


## === cell 2
tf.config.list_physical_devices('GPU')


## === cell 3
import pandas as pd
labels=pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')


## === cell 4
labels.head()


## === cell 5
labels.info()


## === cell 6
labels.describe()


## === cell 7
labels['breed'].value_counts()


## === cell 8
labels['breed'].value_counts().plot(figsize=(25,6),kind='bar');


## === cell 9
labels['breed'].value_counts().mean(),labels['breed'].value_counts().median()


## === cell 10
from IPython.display import Image
Image('/kaggle/input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg')


## === cell 11
import os
os.listdir('/kaggle/input/dog-breed-identification/train')[:10]


## === cell 12
files=['/kaggle/input/dog-breed-identification/train/'+fname+'.jpg' for fname in labels['id']]


## === cell 13
files[:10]


## === cell 14
if len(files)==len(os.listdir('/kaggle/input/dog-breed-identification/train')):
    print('Ye')
else:
    print('Nay')


## === cell 15
Image(files[9000])


## === cell 16
labels.iloc[9000]


## === cell 17
import numpy as np
breeds=labels['breed'].to_numpy()
breeds


## === cell 18
if len(breeds)==len(files):
    print('Ye')
else:
    print('Nay')


## === cell 19
unique_breeds=np.unique(breeds)
len(unique_breeds)


## === cell 20
unique_breeds


## === cell 21
breeds[1],breeds[1]==unique_breeds


## === cell 22
boolean_labels=[breed==unique_breeds for breed in breeds]
boolean_labels


## === cell 23
len(boolean_labels)


## === cell 24
print(breeds[2]) #prints the name of breed from the breed column in the dataframe
print(np.where(breeds[2]==unique_breeds)) #returns intex where True is returned when compared to unique_breeds
print(boolean_labels[2].argmax()) #Highest value in a boolean_label is 1 (True). Returns index of that index
boolean_labels[2].astype(int)


## === cell 25
len(files)


## === cell 26
X=files
y=boolean_labels


## === cell 27
Image(files[13])


## === cell 28
from matplotlib.pyplot import imread
image=imread(files[13])
image


## === cell 29
image.shape


## === cell 30
image[:2]


## === cell 31
tf.constant(image)


## === cell 32
Image(files[0])


## === cell 33
NUM_IMAGES=1000 #@param {type='slider',min=1000,max=10000,step=1000} this works in colab


## === cell 34
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X[:NUM_IMAGES],y[:NUM_IMAGES],train_size=0.8)

len(X_train),len(X_val),len(y_train),len(y_val)


## === cell 35
X_train[:10],y_train[:10]


## === cell 37
IMG_SIZE=224

def process_image(image_path):
    image=tf.io.read_file(image_path) #reading image into some tensor string bullshit
    image=tf.image.decode_jpeg(image,channels=3) #decoding image to tensor matrix w 3 colour channels (R,G,B)
    image=tf.image.convert_image_dtype(image, tf.float32) #normalising 0-255 to 0-1
    image=tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
    
    
    return image 
    


## === cell 38
image_path=files[0]
image=tf.io.read_file(image_path) #reading image
image


## === cell 39
tf.image.decode_jpeg(tf.io.read_file(files[3]),channels=3)


## === cell 40
tf.io.read_file(files[0])


## === cell 41
tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(files[3]),channels=3),tf.float32)


## === cell 42

def get_label(path,label):
    return process_image(path),label


## === cell 43
(process_image(X[13]),tf.constant(y[13]))


## === cell 44
BATCH_SIZE=32

def create_batch(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
    if test_data:
        print("Creating test batches...")
        data=tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch=data.map(process_image).batch(BATCH_SIZE)
        return data_batch
    
    elif valid_data:
        print('Creating validation batches...')
        data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        data_batch=data.map(get_label).batch(BATCH_SIZE)
        return data_batch
    
    else:
        print("Creating train batches")
        data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        data=data.shuffle(len(X))
        data_batch=data.map(get_label).batch(BATCH_SIZE)
        return data_batch
        
        
        
        
        


## === cell 45
data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
list(data)


## === cell 46
train_data=create_batch(X_train,y_train)
val_data=create_batch(X_val,y_val,valid_data=True)


## === cell 47
train_data.element_spec, val_data.element_spec


## === cell 48
import matplotlib.pyplot as plt

def show25img(images,labels):
    plt.figure(figsize=(10,10))
    
    for i in range (25):
        ax=plt.subplot(5,5,i+1) #(rows,columns,index)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels[i].argmax()])
        plt.axis("off")


## === cell 49
train_img,train_label=next(train_data.as_numpy_iterator())
show25img(train_img,train_label) #shuffles everytime I run this... why?


## === cell 50
val_img,val_label=next(val_data.as_numpy_iterator())
show25img(val_img,val_label)


## === cell 51
IMG_SIZE


## === cell 52
INPUT_SHAPE=[0,IMG_SIZE,IMG_SIZE,3]
OUTPUT_SHAPE=len(unique_breeds)
MODEL_URL="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


## === cell 53
def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model=MODEL_URL):
    print("Building model with ",MODEL_URL)
    
    model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
                                 tf.keras.layers.Dense(units=OUTPUT_SHAPE,activation="softmax") # Layer 2 (output layer)
                                 ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    
    model.build(INPUT_SHAPE)
    
    return model


## === cell 54
model=create_model()
model.summary()


## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3494382007.py in <cell line: 0>()
----> 1 model=create_model()
      2 model.summary()

/tmp/ipykernel_11/1185435520.py in create_model(input_shape, output_shape, model)
      3 
      4     # Set up model layers
----> 5     model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
      6                                  tf.keras.layers.Dense(units=OUTPUT_SHAPE,activation="softmax") # Layer 2 (output layer)
      7                                  ])

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f08201e2e50> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 55
os.makedirs("/kaggle/working/logs")


## === cell 56
!ls


## === cell 57
%load_ext tensorboard
import datetime

def tensorboard_callback():
    logdir = os.path.join("drive/My Drive/Data/logs",
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))    
    return tf.keras.callbacks.TensorBoard(logdir)


## === cell 58
early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)


## === cell 59
print("Ye" if tf.config.list_physical_devices("GPU") else "Nay")


## === cell 60
NUM_EPOCHS=100
def train_model():
    model=create_model()
    tb=tensorboard_callback()
    
    model.fit(x=train_data,epochs=NUM_EPOCHS,validation_data=val_data,validation_freq=1,callbacks=[tb,early_stop])
    
    return model


## === cell 61
model=train_model()


## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3707880882.py in <cell line: 0>()
----> 1 model=train_model()

/tmp/ipykernel_11/574296720.py in train_model()
      1 NUM_EPOCHS=100
      2 def train_model():
----> 3     model=create_model()
      4     tb=tensorboard_callback()
      5 

/tmp/ipykernel_11/1185435520.py in create_model(input_shape, output_shape, model)
      3 
      4     # Set up model layers
----> 5     model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
      6                                  tf.keras.layers.Dense(units=OUTPUT_SHAPE,activation="softmax") # Layer 2 (output layer)
      7                                  ])

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f07e5d84a10> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 63
predictions = model.predict(val_data,verbose=1)
predictions


## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/746115216.py in <cell line: 0>()
----> 1 predictions = model.predict(val_data,verbose=1)
      2 predictions

NameError: name 'model' is not defined

## === cell 64
predictions.shape


## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2983742165.py in <cell line: 0>()
----> 1 predictions.shape

NameError: name 'predictions' is not defined

## === cell 65
idx=0
print(predictions[idx])
print(f"Max value is: {np.max(predictions[idx])}")
print(f"index of max value is: {np.argmax(predictions[idx])}")
print(f"predicted breed is {unique_breeds[np.argmax(predictions[idx])]}")


## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2617562746.py in <cell line: 0>()
      1 idx=0
----> 2 print(predictions[idx])
      3 print(f"Max value is: {np.max(predictions[idx])}")
      4 print(f"index of max value is: {np.argmax(predictions[idx])}")
      5 print(f"predicted breed is {unique_breeds[np.argmax(predictions[idx])]}")

NameError: name 'predictions' is not defined

## === cell 66
def get_y_val(index):
    return unique_breeds[np.argmax(predictions[index])]


## === cell 67
get_y_val(0)


## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2978111834.py in <cell line: 0>()
----> 1 get_y_val(0)

/tmp/ipykernel_11/1179265313.py in get_y_val(index)
      1 def get_y_val(index):
----> 2     return unique_breeds[np.argmax(predictions[index])]

NameError: name 'predictions' is not defined

## === cell 68
def unbatchify(data):
    images=[]
    labels=[]
    
    for image,label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(label)
    return images,labels
    


## === cell 69
val_img,val_label=unbatchify(val_data)
val_label


## === cell 70
def plot_pred(index):
    y_val=get_y_val(index)
    y_actual=unique_breeds[np.argmax(val_label[index])]
    img=val_img[index]
    y_prob=np.max(predictions[index])
    
    plt.imshow(img)
    plt.axis("off")
    
    if y_val==y_actual:
        color="green"
    else:
        color="red"
            
    plt.title(f"{y_val} {(y_prob*100):.2f}% ({y_actual})",color=color)
    


## === cell 71
plot_pred(13)


## --- ERROR in cell 71, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3235621578.py in <cell line: 0>()
----> 1 plot_pred(13)

/tmp/ipykernel_11/3831742351.py in plot_pred(index)
      1 def plot_pred(index):
----> 2     y_val=get_y_val(index)
      3     y_actual=unique_breeds[np.argmax(val_label[index])]
      4     img=val_img[index]
      5     y_prob=np.max(predictions[index])

/tmp/ipykernel_11/1179265313.py in get_y_val(index)
      1 def get_y_val(index):
----> 2     return unique_breeds[np.argmax(predictions[index])]

NameError: name 'predictions' is not defined

## === cell 72
get_y_val(1)


## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1878110027.py in <cell line: 0>()
----> 1 get_y_val(1)

/tmp/ipykernel_11/1179265313.py in get_y_val(index)
      1 def get_y_val(index):
----> 2     return unique_breeds[np.argmax(predictions[index])]

NameError: name 'predictions' is not defined

## === cell 73
val_label[1]


## === cell 74
!ls


## === cell 75
def save_model(model,suffix=None):
    modeldir=os.path.join("models",datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
    
    model_path=modeldir+"-"+suffix+".h5"
    print(f"saving to {model_path}")
    
    model.save(model_path)
    return model_path

def load_model(model_path):
    print(f"loading from {model_path}")
    
    model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
    
    return model


## === cell 76
model.save("models/1000-mobilenetV2-Adam.h5", overwrite=True)


## --- ERROR in cell 76, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1646434468.py in <cell line: 0>()
      1 # save_model(model,"1000-mobilenetV2-Adam")
----> 2 model.save("models/1000-mobilenetV2-Adam.h5", overwrite=True)

NameError: name 'model' is not defined

## === cell 77
yoyo=load_model("/kaggle/working/models/1000-mobilenetV2-Adam.h5")


## --- ERROR in cell 77, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/340810108.py in <cell line: 0>()
----> 1 yoyo=load_model("/kaggle/working/models/1000-mobilenetV2-Adam.h5")

/tmp/ipykernel_11/2599264695.py in load_model(model_path)
     11     print(f"loading from {model_path}")
     12 
---> 13     model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
     14 
     15     return model

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/working/models/1000-mobilenetV2-Adam.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 78
yoyo.evaluate(val_data)


## --- ERROR in cell 78, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3632740757.py in <cell line: 0>()
----> 1 yoyo.evaluate(val_data)

NameError: name 'yoyo' is not defined

## === cell 79
len(X)


## === cell 80
len(y)


## === cell 81
full_data=create_batch(X,y)
full_data


## === cell 82
full_model=create_model()


## --- ERROR in cell 82, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1382609803.py in <cell line: 0>()
----> 1 full_model=create_model()

/tmp/ipykernel_11/1185435520.py in create_model(input_shape, output_shape, model)
      3 
      4     # Set up model layers
----> 5     model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
      6                                  tf.keras.layers.Dense(units=OUTPUT_SHAPE,activation="softmax") # Layer 2 (output layer)
      7                                  ])

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f07e4463490> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 83
full_model_tb=tensorboard_callback()
full_model_stop=tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=5)


## === cell 84
full_model.fit(x=full_data,epochs=NUM_EPOCHS,callbacks=[full_model_tb,full_model_stop])


## --- ERROR in cell 84, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4148777874.py in <cell line: 0>()
----> 1 full_model.fit(x=full_data,epochs=NUM_EPOCHS,callbacks=[full_model_tb,full_model_stop])

NameError: name 'full_model' is not defined

## === cell 85
full_model.save("models/full_trained_mobilenetV2-Adam.h5", overwrite=True)


## --- ERROR in cell 85, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/808025476.py in <cell line: 0>()
      1 # save_model(full_model,"full_trained-mobilenetV2-Adam")
----> 2 full_model.save("models/full_trained_mobilenetV2-Adam.h5", overwrite=True)

NameError: name 'full_model' is not defined

## === cell 86
test_path="/kaggle/input/dog-breed-identification/test/"
test_file=[test_path + fname for fname in os.listdir(test_path)]


## === cell 87
test_file[:10]


## === cell 88
len(test_file)


## === cell 89
test_data=create_batch(test_file,test_data=True)


## === cell 90
test_data


## === cell 91
test_predictions=full_model.predict(test_data,verbose=1)


## --- ERROR in cell 91, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1985698972.py in <cell line: 0>()
----> 1 test_predictions=full_model.predict(test_data,verbose=1)

NameError: name 'full_model' is not defined

## === cell 92
test_predictions[:10]


## --- ERROR in cell 92, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3421437898.py in <cell line: 0>()
----> 1 test_predictions[:10]

NameError: name 'test_predictions' is not defined

## === cell 93
test_predictions.shape


## --- ERROR in cell 93, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1029057476.py in <cell line: 0>()
----> 1 test_predictions.shape

NameError: name 'test_predictions' is not defined

## === cell 94
import pandas as pd
preds_df=pd.DataFrame(columns=["id"]+list(unique_breeds))
preds_df


## === cell 95
test_ids=[fname[:-4] for fname in os.listdir("/kaggle/input/dog-breed-identification/test")]


## === cell 96
test_ids[:10]


## === cell 97
preds_df["id"]=test_ids
preds_df.head()


## === cell 98
preds_df[list(unique_breeds)]=test_predictions
preds_df.head()


## --- ERROR in cell 98, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/166925266.py in <cell line: 0>()
----> 1 preds_df[list(unique_breeds)]=test_predictions
      2 preds_df.head()

NameError: name 'test_predictions' is not defined

## === cell 99
preds_df.to_csv("submission.csv",index=False)


## === cell 100
preds_df.shape


## --- ERROR in outputing the csv:
Invalid submission: Submission should be the same length as the answers
