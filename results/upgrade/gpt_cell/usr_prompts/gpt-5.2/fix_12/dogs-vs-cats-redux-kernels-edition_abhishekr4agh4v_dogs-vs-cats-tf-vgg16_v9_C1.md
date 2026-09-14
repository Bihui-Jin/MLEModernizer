# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os as _os

_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<4"])

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os


## === cell 2
! unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip


## === cell 3
! unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip


## === cell 4
train_candidates = [
    "train",
    os.path.join("dogs-vs-cats-redux-kernels-edition", "train"),
    os.path.join("dogs-vs-cats-redux-kernels-edition", "train", "train"),
    os.path.join("/kaggle/working", "train"),
    os.path.join("/kaggle/working", "dogs-vs-cats-redux-kernels-edition", "train"),
    os.path.join(
        "/kaggle/working", "dogs-vs-cats-redux-kernels-edition", "train", "train"
    ),
]
test_candidates = [
    "test",
    os.path.join("dogs-vs-cats-redux-kernels-edition", "test"),
    os.path.join("dogs-vs-cats-redux-kernels-edition", "test", "test"),
    os.path.join("/kaggle/working", "test"),
    os.path.join("/kaggle/working", "dogs-vs-cats-redux-kernels-edition", "test"),
    os.path.join(
        "/kaggle/working", "dogs-vs-cats-redux-kernels-edition", "test", "test"
    ),
]

train_dir = next((p for p in train_candidates if os.path.isdir(p)), None)
test_dir = next((p for p in test_candidates if os.path.isdir(p)), None)

if train_dir is None or test_dir is None:
    raise FileNotFoundError(
        f"Could not find extracted train/test directories. "
        f"Checked train: {train_candidates} ; test: {test_candidates}. "
        f"Current working dir: {os.getcwd()}"
    )

datasets_train = os.listdir(train_dir)
datasets_test = os.listdir(test_dir)


## === cell 5
datasets_train[0:10]  # data are images with this name  ok
datasets_test[0:5]


## === cell 6
labels = [] 
for imagename in datasets_train:
    if 'dog' in  imagename:
        labels.append('dog')
    elif 'cat' in imagename:
        labels.append('cat')


## === cell 7
train_images = []
train_labels = []
for imagename in datasets_train:
    if not isinstance(imagename, str):
        continue
    name_lower = imagename.lower()
    if not name_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue
    if "dog" in name_lower:
        train_images.append(imagename)
        train_labels.append("dog")
    elif "cat" in name_lower:
        train_images.append(imagename)
        train_labels.append("cat")

dfx = pd.DataFrame()
dfx["imagename"] = train_images
dfx["labels"] = train_labels

dftest = pd.DataFrame()
dftest["image"] = datasets_test


## === cell 8
dftest.head()
dfx.head()


## === cell 9
def show_image(imageadd):
    image = cv2.imread(imageadd)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title("imagename")
    plt.imshow(image)


if len(dfx) == 0:
    train_images = []
    train_labels = []
    for root, _, files in os.walk(train_dir):
        for fname in files:
            if not isinstance(fname, str):
                continue
            name_lower = fname.lower()
            if not name_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            rel_path = os.path.relpath(os.path.join(root, fname), start=train_dir)
            rel_lower = rel_path.lower()

            if "dog" in rel_lower:
                train_images.append(rel_path)
                train_labels.append("dog")
            elif "cat" in rel_lower:
                train_images.append(rel_path)
                train_labels.append("cat")

    dfx = pd.DataFrame({"imagename": train_images, "labels": train_labels})

if len(dfx) == 0:
    raise ValueError("No training images were found in train_dir; dfx is empty.")

show_image(os.path.join(train_dir, dfx["imagename"].iloc[0]))


## === cell 10
plt.figure(figsize=(20, 20))
for i in range(10):
    img_path = os.path.join(train_dir, dfx.loc[i, "imagename"])
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 5, i + 1)
    plt.title(dfx.loc[i, "imagename"])
    plt.imshow(image)


## === cell 11
dfx.describe()
dfx.value_counts('labels')
dfx.duplicated('imagename').sum()


## === cell 12
images = []
for imagename in os.listdir(train_dir):
    if not isinstance(imagename, str):
        continue
    name_lower = imagename.lower()
    if not name_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    img_path = os.path.join(train_dir, imagename)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)


## === cell 13
len(images)


## === cell 14
x = np.array(images)
listshape=[]

for i in range(len(x)):
    listshape.append(x[i].shape)
shapearray = np.array(listshape)
shapearray[0:5]


## === cell 15
shapearray.mean(axis=0)


## === cell 16
count = 0
for i in range(len(shapearray)):
    if shapearray[i,2] !=3:
        count = count+1
count


## === cell 17
idg = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                       preprocessing_function=tf.keras.applications.vgg16.preprocess_input ,
                                                      width_shift_range=0.1,
                                                      height_shift_range=0.1,
                                                      zoom_range=0.2,
                                                      rotation_range=25,
                                                     )


## === cell 18
os.mkdir('viewaugimages')


## === cell 19
dfx.imagename[8]


## === cell 20
image = dfx.imagename[8]
img_path = os.path.join(train_dir, image)

image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Could not read image at path: {img_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = np.expand_dims(image, axis=0)
image2.shape


## === cell 21
i=0
for _ in idg.flow(image2,save_to_dir='viewaugimages'):
    i=i+1
    if i >24:
        break


## === cell 22
auglist=[]
for imagename in os.listdir('viewaugimages'):
    image = cv2.imread('viewaugimages/'+ imagename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    auglist.append(image)


## === cell 23
len(auglist)


## === cell 24
plt.figure(figsize= (20,20))
for i in range(25):
    image = auglist[i]
    plt.subplot(5,5,i+1)
    plt.imshow(image)
    


## === cell 25
bs = 32


## === cell 26
train_idg = idg.flow_from_dataframe(dfx,directory = '/kaggle/working/train/',
                                    x_col = 'imagename',y_col = 'labels',
                                    target_size =(180,200),
                                    batch_size =bs,
                                    subset = 'training')


## === cell 28
VGG16transfer = tf.keras.applications.VGG16(
    include_top=False, input_shape=(180, 200, 3), weights="imagenet"
)

tf.keras.utils.plot_model(VGG16transfer)

for layer in VGG16transfer.layers:
    print(layer.name)

    in_shape = getattr(layer, "input_shape", None)
    if in_shape is None:
        try:
            in_shape = tuple(layer.input.shape)
        except Exception:
            in_shape = None
    print(in_shape)

    out_shape = getattr(layer, "output_shape", None)
    if out_shape is None:
        try:
            out_shape = tuple(layer.output.shape)
        except Exception:
            out_shape = None
    print(out_shape)

    print(layer.trainable)
    layer.trainable = False
    print(layer.trainable)


## === cell 29
flat1 = tf.keras.layers.Flatten() (VGG16transfer.output)
d1 = tf.keras.layers.Dense(32,activation = 'relu') (flat1)
pred = tf.keras.layers.Dense(2,activation = 'softmax') (d1)


model1 = tf.keras.Model(inputs = [VGG16transfer.input], outputs = [pred])

for layer in model1.layers:
    print(layer.trainable)


## === cell 30
model1.summary()


## === cell 31
model1.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0031415) ,
              loss = tf.keras.losses.categorical_crossentropy ,
              metrics=['acc']
             )


## === cell 32
train_idg = idg.flow_from_dataframe(
    dfx,
    directory=train_dir,
    x_col="imagename",
    y_col="labels",
    target_size=(180, 200),
    batch_size=bs,
    class_mode="categorical",
    shuffle=True,
)

model1.fit(train_idg, epochs=1, batch_size=bs)


## === cell 33
dict = model1.history.history


## === cell 34
plt.figure(figsize=(15,10))
plt.plot(dict['acc'])
plt.show()


## === cell 35
plt.figure(figsize=(15,10))
plt.plot(dict.get('loss'))
plt.show()


## === cell 36
dftest.sample(2)


## === cell 37
idg2 = tf.keras.preprocessing.image.ImageDataGenerator(
                                                         preprocessing_function=tf.keras.applications.vgg16.preprocess_input
                                                           )
test_gen = idg2.flow_from_dataframe(
    dftest, 
    '/kaggle/working/test', 
    x_col='image',
    class_mode= None,
    target_size=(180,200),
    batch_size=bs,
    shuffle=False
)


## === cell 38
predict = model1.predict(test_gen,batch_size=bs,max_queue_size=1, verbose = 1)


## === cell 39
p=4
print((predict[4,1]))#prob of being a dog
show_image('/kaggle/working/test/'+ dftest.image[p])


## === cell 40
labels_test = (predict[:,1])
labels_test[0:5]


## === cell 41
result = pd.DataFrame()
result['id']=   [i for i in range(1,len(dftest)+1)]
result['label']=labels_test
result.head()


## === cell 43
result.to_csv('submission.csv',index = False)
