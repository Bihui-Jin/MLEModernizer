# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os


## === cell 1
label=pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv') # Loading data
label.head()


## === cell 2
label.has_cactus.value_counts()


## === cell 3
sns.countplot(label,x='has_cactus') # Checking class imbalance


## === cell 4
! unzip -q /kaggle/input/aerial-cactus-identification/train.zip


## === cell 5
! unzip -q /kaggle/input/aerial-cactus-identification/test.zip


## === cell 6
train_dir='/kaggle/working/train/'
test_dir='/kaggle/working/test/'


## === cell 7
_candidates = [
    train_dir,  # keep original if it exists
    "/kaggle/working/aerial-cactus-identification/train/",
    "/kaggle/working/aerial-cactus-identification/aerial-cactus-identification/train/",
    "/kaggle/input/aerial-cactus-identification/train/",
]

for _p in _candidates:
    if os.path.isdir(_p):
        train_dir = _p
        break

len(os.listdir(train_dir))


## === cell 8
os.listdir('/kaggle/working/train')[2]


## === cell 9
r=np.random.randint(1,17500,16)
plt.figure(figsize=(16,16))
for i,v in enumerate(r):
  plt.subplot(4,4,i+1)
  image=cv2.imread(train_dir+os.listdir(train_dir)[v])
  plt.title(label[label['id']==os.listdir(train_dir)[v]]['has_cactus'].values[0])
  image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  plt.imshow(image)


## === cell 10
! pip install tensorflow


## === cell 11
import tensorflow as tf


## === cell 12
idg=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0,
                                                    width_shift_range=0,
                                                    height_shift_range=0,
                                                    horizontal_flip=False,
                                                    vertical_flip=False,
                                                    validation_split=0.1,
                                                    brightness_range=(0,1),
                                                    channel_shift_range=12.5,
                                                    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)


## === cell 13
for i in range(label.shape[0]):
  if label.iloc[i,-1]==1:
    label.iloc[i,-1]='yes'
  else:
    label.iloc[i,-1]='no'


## === cell 14
label.sample(5)


## === cell 15
b=32

train_idg=idg.flow_from_dataframe(label,train_dir,x_col='id',y_col='has_cactus',target_size=(32,32),batch_size=b,
                                  subset='training')

val_idg=idg.flow_from_dataframe(label,train_dir,x_col='id',y_col='has_cactus',target_size=(32,32),batch_size=b,
                                  subset='validation')


## === cell 16
vgg=tf.keras.applications.VGG16(include_top=False,input_shape=(32,32,3),pooling='same')


## === cell 17
vgg.trainable=False


## === cell 18
flat=tf.keras.layers.Flatten(name='FlattenLayer') (vgg.output)
dropout=tf.keras.layers.Dropout(0.3,name='DropoutLayer') (flat)
dense1=tf.keras.layers.Dense(512,name='HiddenLayer1',activation='relu') (dropout)
dense2=tf.keras.layers.Dense(256,name='HiddenLayer2',activation='relu') (dense1)
output=tf.keras.layers.Dense(2,name='OutputLayer',activation='softmax') (dense2)

model=tf.keras.models.Model(inputs=[vgg.input],outputs=output)

model.summary()


## === cell 19
pos=label['has_cactus'].value_counts().values[0]
neg=label['has_cactus'].value_counts().values[1]
total=pos+neg
class_weight_pos=(1/pos)*(total/2.0)
class_weight_neg=(1/neg)*(total/2.0)
print(class_weight_pos,class_weight_neg)
class_weight1={1:class_weight_pos,0:class_weight_neg}


## === cell 20
model.compile(optimizer=tf.keras.optimizers.SGD(),loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.AUC(100,'ROC',name='AUC'),'acc'])


## === cell 21
model.fit(train_idg,batch_size=b,epochs=15,validation_data=val_idg,class_weight=class_weight1)


## === cell 22
train_idg.class_indices


## === cell 23
plt.figure(figsize=(10,5))
plt.subplot(121)
sns.lineplot(model.history.history['acc'],label='Train Acc')
sns.lineplot(model.history.history['val_acc'],label='Validation Acc')
plt.legend()
plt.title('Training & Validation Accuracy')

plt.subplot(122)
sns.lineplot(model.history.history['loss'],label='Train Loss')
sns.lineplot(model.history.history['val_loss'],label='Validation Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.show()


## === cell 24
len(os.listdir(test_dir))


## === cell 25
q=np.random.randint(1,4000,8)
plt.figure(figsize=(16,10))
for i,v in enumerate(q):
  plt.subplot(2,4,i+1)
  image=cv2.imread(test_dir+os.listdir(test_dir)[v])
  image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  plt.title(v)
  plt.imshow(image)


## === cell 26
test_result=pd.DataFrame(os.listdir(test_dir),columns=['id'])
test_result.head()


## === cell 27
test_idg=idg.flow_from_dataframe(test_result,test_dir,batch_size=1,x_col='id',
                                target_size=(32,32),class_mode=None,shuffle=False)


## === cell 28
test_pred=model.predict(test_idg,steps=len(test_idg.filenames))


## === cell 29
test_pred.shape


## === cell 30
test_pred[0:4]


## === cell 31
test_prob=test_pred[:,1]


## === cell 32
test_prob.shape


## === cell 33
test_result['has_cactus']=test_prob


## === cell 34
test_result.sample(5)


## === cell 35
test_result.to_csv('submission.csv',index=False)


## === cell 36
df=pd.read_csv('submission.csv')


## === cell 37
df
