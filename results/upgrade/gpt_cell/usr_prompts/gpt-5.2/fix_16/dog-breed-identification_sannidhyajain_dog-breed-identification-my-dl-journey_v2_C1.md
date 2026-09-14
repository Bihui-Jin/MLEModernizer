# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.13

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

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
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model=MODEL_URL):
    print("Building model with ", MODEL_URL)

    hub_layer = hub.KerasLayer(MODEL_URL)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(lambda x: hub_layer(x)),
            tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.build(INPUT_SHAPE)
    return model


model = create_model()
model.summary()


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


## === cell 63
predictions = model.predict(val_data,verbose=1)
predictions


## === cell 64
predictions.shape


## === cell 65
idx=0
print(predictions[idx])
print(f"Max value is: {np.max(predictions[idx])}")
print(f"index of max value is: {np.argmax(predictions[idx])}")
print(f"predicted breed is {unique_breeds[np.argmax(predictions[idx])]}")


## === cell 66
def get_y_val(index):
    return unique_breeds[np.argmax(predictions[index])]


## === cell 67
get_y_val(0)


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


## === cell 72
get_y_val(1)


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
import os

os.makedirs("models", exist_ok=True)

model.save("models/1000-mobilenetV2-Adam.h5", overwrite=True)


## --- ERROR in cell 76, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2251603800.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;31m# Keras 3 deprecates `save_format`; format is inferred from the file extension.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;31m# Save as .h5 to match the path expected by the next cell's `load_model(...)`.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mmodel[0m[0;34m.[0m[0msave[0m[0;34m([0m[0;34m"models/1000-mobilenetV2-Adam.h5"[0m[0;34m,[0m [0moverwrite[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_list[0;34m(x, memo, deepcopy)[0m
[1;32m    204[0m     [0mappend[0m [0;34m=[0m [0my[0m[0;34m.[0m[0mappend[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m     [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 206[0;31m         [0mappend[0m[0;34m([0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    208[0m [0md[0m[0;34m[[0m[0mlist[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_list[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_tuple[0;34m(x, memo, deepcopy)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_tuple[0;34m(x, memo, deepcopy)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_tuple[0;34m(x, memo, deepcopy)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_list[0;34m(x, memo, deepcopy)[0m
[1;32m    204[0m     [0mappend[0m [0;34m=[0m [0my[0m[0;34m.[0m[0mappend[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m     [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 206[0;31m         [0mappend[0m[0;34m([0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    208[0m [0md[0m[0;34m[[0m[0mlist[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_list[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_tuple[0;34m(x, memo, deepcopy)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    209[0m [0;34m[0m[0m
[1;32m    210[0m [0;32mdef[0m [0m_deepcopy_tuple[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0mdeepcopy[0m[0;34m=[0m[0mdeepcopy[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 211[0;31m     [0my[0m [0;34m=[0m [0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    212[0m     [0;31m# We're not going to put the tuple in the memo, but it's still important we[0m[0;34m[0m[0;34m[0m[0m
[1;32m    213[0m     [0;31m# check for it, in case the tuple contains recursive mutable structures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    151[0m             [0mcopier[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mx[0m[0;34m,[0m [0;34m"__deepcopy__"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    152[0m             [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 153[0;31m                 [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    154[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    155[0m                 [0mreductor[0m [0;34m=[0m [0mdispatch_table[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_list[0;34m(x, memo, deepcopy)[0m
[1;32m    204[0m     [0mappend[0m [0;34m=[0m [0my[0m[0;34m.[0m[0mappend[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m     [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 206[0;31m         [0mappend[0m[0;34m([0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    208[0m [0md[0m[0;34m[[0m[0mlist[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_list[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_list[0;34m(x, memo, deepcopy)[0m
[1;32m    204[0m     [0mappend[0m [0;34m=[0m [0my[0m[0;34m.[0m[0mappend[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m     [0;32mfor[0m [0ma[0m [0;32min[0m [0mx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 206[0;31m         [0mappend[0m[0;34m([0m[0mdeepcopy[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    208[0m [0md[0m[0;34m[[0m[0mlist[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_list[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    170[0m                     [0my[0m [0;34m=[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    171[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 172[0;31m                     [0my[0m [0;34m=[0m [0m_reconstruct[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m,[0m [0;34m*[0m[0mrv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    173[0m [0;34m[0m[0m
[1;32m    174[0m     [0;31m# If is its own copy, don't memoize.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_reconstruct[0;34m(x, memo, func, args, state, listiter, dictiter, deepcopy)[0m
[1;32m    269[0m     [0;32mif[0m [0mstate[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    270[0m         [0;32mif[0m [0mdeep[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 271[0;31m             [0mstate[0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mstate[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    272[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0my[0m[0;34m,[0m [0;34m'__setstate__'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m             [0my[0m[0;34m.[0m[0m__setstate__[0m[0;34m([0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    144[0m     [0mcopier[0m [0;34m=[0m [0m_deepcopy_dispatch[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mcls[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m     [0;32mif[0m [0mcopier[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0my[0m [0;34m=[0m [0mcopier[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0;32mif[0m [0missubclass[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36m_deepcopy_dict[0;34m(x, memo, deepcopy)[0m
[1;32m    229[0m     [0mmemo[0m[0;34m[[0m[0mid[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m     [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0mx[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0my[0m[0;34m[[0m[0mdeepcopy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0mdeepcopy[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mmemo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m     [0;32mreturn[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m [0md[0m[0;34m[[0m[0mdict[0m[0;34m][0m [0;34m=[0m [0m_deepcopy_dict[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/copy.py[0m in [0;36mdeepcopy[0;34m(x, memo, _nil)[0m
[1;32m    159[0m                     [0mreductor[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mx[0m[0;34m,[0m [0;34m"__reduce_ex__"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m                     [0;32mif[0m [0mreductor[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 161[0;31m                         [0mrv[0m [0;34m=[0m [0mreductor[0m[0;34m([0m[0;36m4[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    162[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    163[0m                         [0mreductor[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mx[0m[0;34m,[0m [0;34m"__reduce__"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: cannot pickle 'FuncGraph' object

## === cell 77
yoyo=load_model("/kaggle/working/models/1000-mobilenetV2-Adam.h5")
