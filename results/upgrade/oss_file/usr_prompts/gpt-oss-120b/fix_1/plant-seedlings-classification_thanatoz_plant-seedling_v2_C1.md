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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.6146

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import seaborn as sns

import os
print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 33
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


## === cell 2
root = '../input/train/Maize/3a6d4d007.png'
img = cv2.imread(root)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = cv2.resize(img,(128,128))
image_segmented = segment_plant(img)
image_sharpen = sharpen_image(image_segmented)
plt.imshow(image_sharpen)


## === cell 3
    root = '../input/train'
    folders = os.listdir(root)
    X = []
    Y = []
    names={}
    ptr = 0
    for folder in  folders:
        names[ptr]=folder
        files = os.listdir(os.path.join(root,folder))
        for file in files:
            image_path = os.path.join(os.path.join(root,folder,file))
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_segmented = segment_plant(img)
            image_sharpen = sharpen_image(image_segmented)
            img = cv2.resize(image_sharpen,(128,128))
            img=img/255
            X.append(img)
            Y.append(ptr)
        ptr+=1

    X = np.array(X)
    Y = np.array(Y)
    names    


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/2877783570.py in <cell line: 0>()
     11         image_path = os.path.join(os.path.join(root,folder,file))
     12         img = cv2.imread(image_path)
---> 13         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
     14         image_segmented = segment_plant(img)
     15         image_sharpen = sharpen_image(image_segmented)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'


## === cell 4
def display_dataset(X,Y, h=128, w=128, rows=5, cols=2, display_labels=True):
    f, ax = plt.subplots(cols, rows)
    for i in range(rows):
        for j in range(cols):
            index=np.random.randint(0,X.shape[0])
            ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
            ax[j,i].set_title(Y[index])
    plt.xticks()
    plt.show()


## === cell 5
X = X.reshape(X.shape[0],-1)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/29146084.py in <cell line: 0>()
----> 1 X = X.reshape(X.shape[0],-1)

AttributeError: 'list' object has no attribute 'reshape'

## === cell 6
X.shape


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/106882318.py in <cell line: 0>()
----> 1 X.shape

AttributeError: 'list' object has no attribute 'shape'

## === cell 7
display_dataset(X, Y)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1242111239.py in <cell line: 0>()
----> 1 display_dataset(X, Y)

/tmp/ipykernel_11/3901918339.py in display_dataset(X, Y, h, w, rows, cols, display_labels)
      3     for i in range(rows):
      4         for j in range(cols):
----> 5             index=np.random.randint(0,X.shape[0])
      6             ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
      7             ax[j,i].set_title(Y[index])

AttributeError: 'list' object has no attribute 'shape'

## === cell 8
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=True, test_size=0.1)


## === cell 9
display_dataset(X_train, Y_train)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/352359007.py in <cell line: 0>()
----> 1 display_dataset(X_train, Y_train)

/tmp/ipykernel_11/3901918339.py in display_dataset(X, Y, h, w, rows, cols, display_labels)
      3     for i in range(rows):
      4         for j in range(cols):
----> 5             index=np.random.randint(0,X.shape[0])
      6             ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
      7             ax[j,i].set_title(Y[index])

AttributeError: 'list' object has no attribute 'shape'

## === cell 10
display_dataset(X_test, Y_test)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2723932836.py in <cell line: 0>()
----> 1 display_dataset(X_test, Y_test)

/tmp/ipykernel_11/3901918339.py in display_dataset(X, Y, h, w, rows, cols, display_labels)
      3     for i in range(rows):
      4         for j in range(cols):
----> 5             index=np.random.randint(0,X.shape[0])
      6             ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
      7             ax[j,i].set_title(Y[index])

AttributeError: 'list' object has no attribute 'shape'

## === cell 11
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    
    return X, Y


## === cell 12
def initialize_parameters():
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [50, 49152], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [50, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [15, 50],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [15, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [12, 15], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [12, 1], initializer=tf.zeros_initializer())
   
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


## === cell 13
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2) # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3) # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


## === cell 14

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    print(logits, labels)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost


## === cell 15
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


## === cell 16
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


## === cell 17
def predict(X, parameters, pred_val=1):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [49152, pred_val])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


## === cell 18
def forward_propagation_for_predict(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


## === cell 19
def model(train_X, train_Y, test_X, test_Y, learning_rate = 0.0001,
          num_epochs = 1000, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    """
    
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)      # to keep consistent results
    seed = 3                   # to keep consistent results
    (n_x, m) = train_X.shape   # (n_x: input size, m : number of examples in the train set)
    n_y = train_Y.shape[0]     # n_y : output size
    costs = []                 # To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(train_X, train_Y, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, Y: minibatch_Y})

            epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: train_X, Y: train_Y}))
        print ("Test Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))
        
        return parameters


## === cell 20
Y_train = convert_to_one_hot(Y_train, 12)
Y_test = convert_to_one_hot(Y_test, 12)
print(Y_train.shape, Y_test.shape)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1909670032.py in <cell line: 0>()
----> 1 Y_train = convert_to_one_hot(Y_train, 12)
      2 Y_test = convert_to_one_hot(Y_test, 12)
      3 print(Y_train.shape, Y_test.shape)

/tmp/ipykernel_11/1903894435.py in convert_to_one_hot(Y, C)
      1 def convert_to_one_hot(Y, C):
----> 2     Y = np.eye(C)[Y.reshape(-1)].T
      3     return Y

AttributeError: 'list' object has no attribute 'reshape'

## === cell 21
X_train=X_train.reshape(X_train.shape[0],-1).T
X_test=X_test.reshape(X_test.shape[0],-1).T
print(X_train.shape, X_test.shape)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/615893140.py in <cell line: 0>()
----> 1 X_train=X_train.reshape(X_train.shape[0],-1).T
      2 X_test=X_test.reshape(X_test.shape[0],-1).T
      3 print(X_train.shape, X_test.shape)

AttributeError: 'list' object has no attribute 'reshape'

## === cell 23
parameters1 = model(X_train, Y_train, X_test, Y_test, learning_rate=0.001, num_epochs=100)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2698456547.py in <cell line: 0>()
----> 1 parameters1 = model(X_train, Y_train, X_test, Y_test, learning_rate=0.001, num_epochs=100)

/tmp/ipykernel_11/3857673744.py in model(train_X, train_Y, test_X, test_Y, learning_rate, num_epochs, minibatch_size, print_cost)
      6 
      7     ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
----> 8     tf.set_random_seed(1)      # to keep consistent results
      9     seed = 3                   # to keep consistent results
     10     (n_x, m) = train_X.shape   # (n_x: input size, m : number of examples in the train set)

AttributeError: module 'tensorflow' has no attribute 'set_random_seed'

## === cell 24
root = '../input/train/Maize/3a6d4d007.png'
imag = cv2.imread(root)
imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
imag = cv2.resize(imag,(128,128))
image_segmented = segment_plant(imag)
image_sharpen = sharpen_image(image_segmented)
imag = cv2.resize(image_sharpen,(128,128))
imag = imag/255

imb = imag.reshape(1, 128*128*3).T
my_image_prediction = predict(imb, parameters1)
plt.imshow(imb.reshape(128,128,3))
print("Your algorithm predicts: y = " + str(names[int(np.squeeze(my_image_prediction))]))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/905895001.py in <cell line: 0>()
      9 
     10 imb = imag.reshape(1, 128*128*3).T
---> 11 my_image_prediction = predict(imb, parameters1)
     12 plt.imshow(imb.reshape(128,128,3))
     13 print("Your algorithm predicts: y = " + str(names[int(np.squeeze(my_image_prediction))]))

NameError: name 'parameters1' is not defined

## === cell 25
root = '../input/test/'
files = os.listdir(root)
x=[]
y=[]
for file in files:
    y.append(file)
    imag = cv2.imread(os.path.join(root,file))
    imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
    imag = cv2.resize(imag,(128,128))
    image_segmented = segment_plant(imag)
    imag = sharpen_image(image_segmented)
    imag = imag/255
    imb = imag.reshape(-1)
    x.append(imb)
x=np.array(x)
y=np.array(y)
x = x.reshape(x.shape[0],-1).T
print(x.shape, y.shape)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/3384354208.py in <cell line: 0>()
      6     y.append(file)
      7     imag = cv2.imread(os.path.join(root,file))
----> 8     imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
      9     imag = cv2.resize(imag,(128,128))
     10     image_segmented = segment_plant(imag)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'


## === cell 26
preda = predict(x, parameters1, x.shape[1])


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2757412865.py in <cell line: 0>()
----> 1 preda = predict(x, parameters1, x.shape[1])

NameError: name 'parameters1' is not defined

## === cell 27
import pickle
pickle.dump(parameters1, open('parameters_pickle.p','wb'))
pickle.dump(names, open('parameters_pickle.p','wb'))


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/922133041.py in <cell line: 0>()
      1 import pickle
----> 2 pickle.dump(parameters1, open('parameters_pickle.p','wb'))
      3 pickle.dump(names, open('parameters_pickle.p','wb'))

NameError: name 'parameters1' is not defined

## === cell 28
put = [names[i] for i in preda]
put


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805512177.py in <cell line: 0>()
----> 1 put = [names[i] for i in preda]
      2 put

NameError: name 'preda' is not defined

## === cell 29
res = pd.DataFrame({'file':y, 'species':put })
res.head()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2838873240.py in <cell line: 0>()
----> 1 res = pd.DataFrame({'file':y, 'species':put })
      2 res.head()

NameError: name 'put' is not defined

## === cell 30
res.to_csv('submission.csv', index=False)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/206042853.py in <cell line: 0>()
----> 1 res.to_csv('submission.csv', index=False)

NameError: name 'res' is not defined
