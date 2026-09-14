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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.7

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
PyYAML==6.0.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.28311

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import glob, os, yaml, csv
import numpy as np
from cv2 import imread , resize
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Flatten, Reshape, Dense,Conv2DTranspose
from keras import Model, callbacks
import matplotlib.pyplot as plt
%matplotlib inline


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
X=np.array([imread(each) for each in glob.glob(os.path.join(os.getcwd() , '../input/train/*.png'))[:4]])
y=np.array([imread(each) for each in glob.glob(os.path.join(os.getcwd() , '../input/train_cleaned/*.png'))[:4]])


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3432699358.py in <cell line: 0>()
      1 #We read the images and store them in numpy arrays
      2 X=np.array([imread(each) for each in glob.glob(os.path.join(os.getcwd() , '../input/train/*.png'))[:4]])
----> 3 y=np.array([imread(each) for each in glob.glob(os.path.join(os.getcwd() , '../input/train_cleaned/*.png'))[:4]])

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4,) + inhomogeneous part.

## === cell 2
plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(241+i)
    fig=plt.imshow(X[i])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(245+i)
    fig=plt.imshow(y[i])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
plt.show()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3565743321.py in <cell line: 0>()
      6     fig.axes.get_yaxis().set_visible(False)
      7     plt.subplot(245+i)
----> 8     fig=plt.imshow(y[i])
      9     fig.axes.get_xaxis().set_visible(False)
     10     fig.axes.get_yaxis().set_visible(False)

NameError: name 'y' is not defined

## === cell 3
shapes=np.unique([imread(each).shape for each in glob.glob(os.path.join(os.getcwd() , '../input/train/*.png'))],axis=0)
print(shapes)


## === cell 4
class Autoencoder:
    def __init__(self,
                 dimensions_factor=2,
                 layers=2,
                 k=3,
                 filter_size=None, 
                 pooling_factor=None,
                 only_decoder=False,
                 only_encoder=False,
                 loss='mean_squared_error',
                 channels=3):
        
        if not (only_decoder or only_encoder):
            self.channels=channels
            self.layers=layers
            self.k=k
            self.filter_size=[(k,k)]*(layers*2+1) if (filter_size is None) else filter_size
            self.pooling_factor=[2]*(layers*2) if (pooling_factor is None) else pooling_factor
            self.dimensions_factor=dimensions_factor
        
        if not only_decoder:
            input_img = Input(shape=(None, None, self.channels))
            x=input_img
            for i in range(self.layers):
                filters=int(self.pooling_factor[i]*int(x.shape[-1])*self.dimensions_factor)
                x = Conv2D(filters, self.filter_size[i], activation='relu', padding='same',name='encoding_conv_'+str(i))(x)
                x = MaxPooling2D((self.pooling_factor[i], self.pooling_factor[i]), padding='valid',name='encoding_pool_'+str(i))(x)
            self.code_filters=filters
        else:
            input_img=Input(shape=(None, None, self.code_filters))
            x=input_img
        if not only_encoder:
            for i in range(self.layers,2*self.layers):
                filters=int(int(x.shape[-1])/(self.pooling_factor[i])*self.dimensions_factor)
                x = Conv2DTranspose(filters, self.filter_size[i], activation='relu', padding='same',name='decoding_conv_'+str(i))(x)
                x = UpSampling2D((self.pooling_factor[i], self.pooling_factor[i]),name='decoding_pool_'+str(i))(x)
            
            x = Conv2D(self.channels, (self.filter_size[-1]), activation='sigmoid', padding='same',name='decoded')(x)
        
        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer='adam', loss=loss, metrics=['mse'])
        
        if not (only_decoder or only_encoder):
            self.model=autoencoder
        else:
            return autoencoder

    def get_encoder(self):
        if self.model is None:
            raise('Model is not created')
        else:
            encoder=self.__init__(only_encoder=True)
            for layer in encoder.layers[1:]:
                weights=self.model.get_layer(name=layer.name).get_weights()
                layer.set_weights(weights)
            return encoder

    def get_decoder(self):
        if self.model is None:
            raise('Model is not created')
        else:
            decoder=self.__init__(only_decoder=True)
            for layer in decoder.layers[1:]:
                weights=self.model.get_layer(name=layer.name).get_weights()
                layer.set_weights(weights)
            return decoder


## === cell 5
class Image_generator():
    def __init__(self,path,val_percentage=0.115,batch_size=4, reduction_factor=4):
        self.base_path=path
        self.batch_size=batch_size
        self.reduction_factor=reduction_factor
        
        self.y=np.array([imread(each).astype('float32') / 255 for each in sorted(glob.glob(os.path.join(path, '../input/train_cleaned/*.png')))])
        self.X=np.array([imread(each).astype('float32') / 255 for each in sorted(glob.glob(os.path.join(path , '../input/train/*.png')))])

        self.idx_train,self.idx_val = self.split_batches(val_percentage)
        
        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)
    
    def split_batches(self,val_percentage):
        idx=[]
        self.shapes=np.unique([each.shape for each in self.X],axis=0)
        for shape in self.shapes:
            shape_idx=np.argwhere([all(a.shape==shape) for a in self.X]).flatten()
            np.random.shuffle(shape_idx)
            idx.extend(np.array_split(shape_idx, len(shape_idx)/self.batch_size))
        idx=np.array([x for x in idx if x.size == self.batch_size])
        np.random.shuffle(idx)
        samples=len(idx)
        val_samples=int(samples*val_percentage)
        
        return idx[:-val_samples],idx[-val_samples:]

    def check_size(self,batch_x,batch_y,axis):
        for each_axis in axis:
            while (batch_x.shape[each_axis]%self.reduction_factor)!=0:
                batch_x=np.insert(batch_x,batch_x.shape[each_axis],1,axis=each_axis)
                batch_y=np.insert(batch_y,batch_y.shape[each_axis],1,axis=each_axis)
        return batch_x,batch_y

    def get_train_batch(self):
        while True:
            batch_x = np.stack(self.X[self.idx_train[0]])
            batch_y = np.stack(self.y[self.idx_train[0]])
            batch_x, batch_y = self.check_size(batch_x, batch_y,[-2,-3])
            self.idx_train = np.roll(self.idx_train,1,axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_x = np.stack(self.X[self.idx_val[0]])
            batch_y = np.stack(self.y[self.idx_val[0]])
            batch_x, batch_y = self.check_size(batch_x, batch_y,[-2,-3])
            self.idx_val = np.roll(self.idx_val,1,axis=0)
            yield batch_x, batch_y


## === cell 6
np.random.seed(42)
data_generator=Image_generator(os.getcwd())
autoencoder=Autoencoder(loss='mse')


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/848589330.py in <cell line: 0>()
      1 np.random.seed(42)
----> 2 data_generator=Image_generator(os.getcwd())
      3 autoencoder=Autoencoder(loss='mse')

/tmp/ipykernel_11/82531394.py in __init__(self, path, val_percentage, batch_size, reduction_factor)
      5         self.reduction_factor=reduction_factor
      6 
----> 7         self.y=np.array([imread(each).astype('float32') / 255 for each in sorted(glob.glob(os.path.join(path, '../input/train_cleaned/*.png')))])
      8         self.X=np.array([imread(each).astype('float32') / 255 for each in sorted(glob.glob(os.path.join(path , '../input/train/*.png')))])
      9 

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (115,) + inhomogeneous part.

## === cell 7
log=autoencoder.model.fit_generator(generator = data_generator.get_train_batch(),
                                  steps_per_epoch=data_generator.train_steps,
                                  epochs=1,
                                  shuffle=False,
                                  validation_data = data_generator.get_val_batch(),
                                  validation_steps = data_generator.val_steps,
                                  use_multiprocessing = False)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793935988.py in <cell line: 0>()
----> 1 log=autoencoder.model.fit_generator(generator = data_generator.get_train_batch(),
      2                                   steps_per_epoch=data_generator.train_steps,
      3                                   epochs=1,
      4                                   shuffle=False,
      5                                   validation_data = data_generator.get_val_batch(),

NameError: name 'autoencoder' is not defined

## === cell 8
plt.figure()
[plt.plot(v,label=str(k)) for k,v in log.history.items()]
plt.legend()
plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2806192552.py in <cell line: 0>()
      1 plt.figure()
----> 2 [plt.plot(v,label=str(k)) for k,v in log.history.items()]
      3 plt.legend()
      4 plt.show()

NameError: name 'log' is not defined

## === cell 9
def to_csv(npdata,ids):
    with open('submission.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(('id','value'))
        for i,each in enumerate(npdata):
            rows,cols,_=each.shape
            for row in range(rows):
                for col in range(cols):
                    id_pixel=str(ids[i])+'_'+str(row+1)+'_'+str(col+1)
                    value_pixel=str(np.mean(each[row,col,:]))
                    csvwriter.writerow([id_pixel,value_pixel])

X_test=np.array([imread(each) for each in sorted(glob.glob(os.path.join(os.getcwd() , '../input/test/*.png')))])
ids=[each.split('/')[-1][:-4] for each in sorted(glob.glob(os.path.join(os.getcwd() , '../input/test/*.png')))]
predictions=[]
for x in X_test:
    original_shape=x.shape
    while (x.shape[0]%4)!=0:
        x=np.insert(x,x.shape[0],1,axis=0)
    while (x.shape[1]%4)!=0:
        x=np.insert(x,x.shape[1],1,axis=1)
    x=x.reshape((1,)+x.shape)
    
    prediction=autoencoder.model.predict(x)
    prediction=prediction.reshape(prediction.shape[1:])
    prediction=prediction[:original_shape[0],:original_shape[1],:]
    
    predictions.append(prediction)
    print('%d of %d predictions calculated'%(len(predictions),len(X_test)), end='\r') 
print('\nSaving...')
to_csv(np.array(predictions),ids)
print('Saved')


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/997781236.py in <cell line: 0>()
     12                     csvwriter.writerow([id_pixel,value_pixel])
     13 
---> 14 X_test=np.array([imread(each) for each in sorted(glob.glob(os.path.join(os.getcwd() , '../input/test/*.png')))])
     15 ids=[each.split('/')[-1][:-4] for each in sorted(glob.glob(os.path.join(os.getcwd() , '../input/test/*.png')))]
     16 predictions=[]

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (29,) + inhomogeneous part.

## === cell 10
encoder= autoencoder.get_encoder()

example=X_test[10].reshape((1,)+X_test[10].shape)
filters=encoder.predict(example)

plt.figure(1)
plt.imshow(example.reshape(example.shape[1:4]))


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4248171230.py in <cell line: 0>()
----> 1 encoder= autoencoder.get_encoder()
      2 
      3 example=X_test[10].reshape((1,)+X_test[10].shape)
      4 filters=encoder.predict(example)
      5 

NameError: name 'autoencoder' is not defined

## === cell 11
%%capture
import matplotlib.animation as animation
from IPython.display import HTML

fig=plt.figure(2)
ims=[]
for i in range(filters.shape[-1]):
    im = plt.imshow(filters[:,:,:,i].reshape(filters.shape[1:3]), animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=0)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2987028139.py in <cell line: 0>()
      4 fig=plt.figure(2)
      5 ims=[]
----> 6 for i in range(filters.shape[-1]):
      7     im = plt.imshow(filters[:,:,:,i].reshape(filters.shape[1:3]), animated=True)
      8     ims.append([im])

NameError: name 'filters' is not defined

## === cell 12
HTML(ani.to_jshtml())


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3380411262.py in <cell line: 0>()
----> 1 HTML(ani.to_jshtml())

NameError: name 'ani' is not defined
