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
Predict whether an image contains a ship or an iceberg.

## Metric
Log loss.

## Submission Format
For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1). The file should contain a header and have the following format:

```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
3aa99a38,0.9
etc.
```

## Dataset
The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

The data (`train.json`, `test.json`) is presented in `json` format.

The files consist of a list of images, and for each image, you can find the following fields:

- **id** - the id of the image
- **band_1, band_2** - the [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html) image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are **float** numbers with unit being [dB](https://en.wikipedia.org/wiki/Decibel). Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically).
- **inc_angle** - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

sample_submission.csv: The submission file in the correct format:

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        input/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        working/
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
```

-> data/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/statoil-iceberg-classifier-challenge/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> input/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> (stopped after 10 files for performance)

# 5. Target score

0.398458066134105

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, isfinite, isnan
from itertools import combinations, compress
import cv2
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

def read_jason(file, loc='../input/'):

    df = pd.read_json('{}{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    
    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)
    
    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2
    
    return df, bands

def img_to_stats(paths):
    
    img_id, img = paths[0], paths[1]
    
    np.seterr(divide='ignore', invalid='ignore')
    
    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    
    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])
            
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])] 
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]), (sub_st[-1] / sub_st[1])] #normalized by stdev
            st += sub_st
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()] #blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]
            
            if opt_poly:
                st_interv.append(sub_st)
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]                
 
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))] #only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]
                
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan
                
    except:
        print('except: ')
    
    return [img_id, st]


def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8) #(cpu_count())
    ret = p.map(img_to_stats, paths.items())
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[f] for f in paths.keys()]
    return np.array(fdata, dtype=np.float32)


def process(df, bands):

    data = extract_img_stats({k: v for k, v in zip(df['id'].tolist(), bands)})
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1)

    print(data.shape)
    return data

if __name__ == '__main__':
    
    np.random.seed(1017)
    target_id = 'is_iceberg'
    
    train, train_bands = read_jason(file='train.json', loc='../input/')
    test, test_bands = read_jason(file='test.json', loc='../input/')

    train_X = process(df=train, bands=train_bands)
    train_y = train[target_id].values

    test_X = process(df=test, bands=test_bands)




## === cell 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.fit_transform(test_X)


## === cell 2
def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s
def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])
                    
    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
 

        
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
        
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    
    return A2, cache

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
    
    dZ2 = A2 - Y
    
    
    dW2 = 1./m * np.dot(dZ2, A1.T)  
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)   
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    
    gradients = {  "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
def  backward_propagation_with_regularization(X, Y, cache, lambd) :
    m = X.shape[1]
    
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
    
    dZ2 = A2 - Y
    
    
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    
    gradients = {  
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(i)] = Wi
                    parameters['b' + str(i)] = bi
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(i)] = dWi
                    grads['db' + str(i)] = dbi
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    n = len(parameters) // 2 # number of layers in the neural networks

    for k in range(n):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    a2, caches = forward_propagation(X, parameters)
    
    for i in range(0, a2.shape[1]):
        if a2[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0


    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def compute_cost(a2 ,Y):
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a2),Y) + np.multiply(-np.log(1 - a2), 1 - Y)
    cost = 1./m * np.nansum(logprobs)
    
    return cost
def compute_cost_with_regularization(A2, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
     
    
    cross_entropy_cost = compute_cost(A2, Y) # This gives you the cross-entropy part of the cost
    
    L2_regularization_cost = 1/m * lambd/2 * (np.sum(np.square(W1))+np.sum(np.square(W2)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    a2, cache = forward_propagation(X, parameters)
    return a2


## === cell 3
def model(X, Y, learning_rate = 0.3, num_iterations = 2500, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20,1]
    
    parameters = initialize_parameters(layers_dims)


    for i in range(0, num_iterations):

        if keep_prob == 1:
            a2, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a2, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        if lambd == 0:
            cost = compute_cost(a2, Y)
        else:
            cost = compute_cost_with_regularization(a2, Y, parameters, lambd)
            
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

## === cell 4
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=42)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/650473688.py in <cell line: 0>()
----> 1 from sklearn.cross_validation import train_test_split
      2 X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=42)

ModuleNotFoundError: No module named 'sklearn.cross_validation'

## === cell 5
X=X_train.T
y_train=y_train.reshape(y_train.shape[0],1)
Y=y_train.T


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2602266742.py in <cell line: 0>()
----> 1 X=X_train.T
      2 y_train=y_train.reshape(y_train.shape[0],1)
      3 Y=y_train.T

NameError: name 'X_train' is not defined

## === cell 6


Xts=X_test.T
y_test=y_test.reshape(y_test.shape[0],1)
Yts=y_test.T



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3034023505.py in <cell line: 0>()
----> 1 Xts=X_test.T
      2 y_test=y_test.reshape(y_test.shape[0],1)
      3 Yts=y_test.T
      4 

NameError: name 'X_test' is not defined

## === cell 7
print ("train_x's shape: " + str(X.shape))
print ("test_x's shape: " + str(Xts.shape))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1444034994.py in <cell line: 0>()
----> 1 print ("train_x's shape: " + str(X.shape))
      2 print ("test_x's shape: " + str(Xts.shape))

NameError: name 'X' is not defined

## === cell 8
import matplotlib  
from matplotlib import pyplot as plt
parameters = model(X, Y, lambd = 0.7)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3770643252.py in <cell line: 0>()
      1 import matplotlib
      2 from matplotlib import pyplot as plt
----> 3 parameters = model(X, Y, lambd = 0.7)

NameError: name 'X' is not defined

## === cell 9
def predict(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    probas, caches = forward_propagation(X, parameters)

    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
        
    return probas

predictions_submit = predict(test_X.T, parameters )
sub = pd.DataFrame({'id': test['id'].values, target_id: predictions_submit.T[:,0]})
 
sub.to_csv('L_layer_net_Kaggle_Boy4.csv',index=False)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1807241952.py in <cell line: 0>()
     33     return probas
     34 
---> 35 predictions_submit = predict(test_X.T, parameters )
     36 sub = pd.DataFrame({'id': test['id'].values, target_id: predictions_submit.T[:,0]})
     37 

NameError: name 'parameters' is not defined

## === cell 10
predictions_submit

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2655764531.py in <cell line: 0>()
----> 1 predictions_submit

NameError: name 'predictions_submit' is not defined

## === cell 11
from sklearn.metrics  import log_loss
predict_test=predict(X_test.T, parameters)
log_loss(y_test,predict_test.T)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1210415262.py in <cell line: 0>()
      1 from sklearn.metrics  import log_loss
----> 2 predict_test=predict(X_test.T, parameters)
      3 log_loss(y_test,predict_test.T)

NameError: name 'X_test' is not defined
