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

No external packages required in the script and installed.

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

0.59897

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

path = "../input/"
test = pd.read_json("../input/test.json")
train = pd.read_json("../input/train.json")


## === cell 10
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    
    s = 1/(1+np.exp(-z))
    
    
    return s


## === cell 15
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    
    w = np.zeros(dim).reshape(dim,1)
    b = 0
    

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


## === cell 20
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    J=−1m∑mi=1y(i)log(a(i))+(1−y(i))log(1−a(i))J=−1m∑i=1my(i)log⁡(a(i))+(1−y(i))log⁡(1−a(i))
    
    
    """
    
    
    
    m = X.shape[1]
    
    
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    mat1=(Y*np.log(A))
    mat2=((1-Y)*np.log(1-A))
    
    
    cost = -1/m*np.sum(mat1+mat2)
    
    
    
    dw = 1/m*(np.dot(X, (A-Y).T))
    db = 1/m*np.sum((A-Y))
    

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


## === cell 25
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


## === cell 30
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    
    Y_prediction = np.array(((A > 0.5).squeeze()*1).reshape(1,m))
    
    
    assert(Y_prediction.shape == (1, m))
    
    return (Y_prediction,A)


## === cell 37
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test, A_test = predict(w, b, X_test)
    Y_prediction_train, A_train = predict(w, b, X_train)

    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "train_with_prob" : A_train,
         "test_with_prob" : A_test,
         "num_iterations": num_iterations}
    return d


## === cell 46
def train_test_split_fun(array_in, array_out, split_perc=0.25):
    from sklearn.model_selection import train_test_split
    X_train, X_val_test, y_train, y_val_test = train_test_split(array_in.T, array_out.T,
                                                    stratify=array_out.T, 
                                                     test_size=split_perc)
    dataset = (X_train.T, X_val_test.T, y_train.T, y_val_test.T)
    return dataset


## === cell 51
def JSON_to_array(split_perc = 0.25, file='train.json'):


    train_set=pd.read_json(path+file)
    inc_set = train_set[train_set['inc_angle']!='na']
    
    band_1=[np.array(i) for i in train_set['band_1']]
    band_2=[np.array(i) for i in train_set['band_2']]
    inc_ang = [np.array(i) for i in inc_set['inc_angle']]

    ice_berg=[np.array(i) for i in train_set['is_iceberg']]
    ice_berg = np.array(ice_berg).reshape(1, 1604)
    inc_set_ice_berg = [np.array(i) for i in inc_set['is_iceberg']]
    inc_set_ice_berg = np.array(inc_set_ice_berg).reshape(1, 1471)

    max_band_1 = np.max(np.array(np.abs(band_1)))
    max_band_2 = np.max(np.array(np.abs(band_2)))
    max_inc_ang = np.max(np.array(np.abs(inc_ang)))
    
    band_2 = np.array(band_2).T/max_band_2
    band_1 = np.array(band_1).T/max_band_1
    inc_ang = np.array(inc_ang).T/max_inc_ang
    inc_ang = inc_ang.reshape(1, 1471)
    
    
    band_1_array = train_test_split_fun(band_1, ice_berg, split_perc=split_perc)    
    band_2_array = train_test_split_fun(band_2, ice_berg, split_perc=split_perc)    
    inc_ang_array = train_test_split_fun(inc_ang, inc_set_ice_berg, split_perc=split_perc)    
    
    return(band_1_array, band_2_array, inc_ang_array)


## === cell 55
(band_1_array, band_2_array, inc_ang_array) = JSON_to_array(file='train.json')


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1995091355.py in <cell line: 0>()
----> 1 (band_1_array, band_2_array, inc_ang_array) = JSON_to_array(file='train.json')

/tmp/ipykernel_11/1631624718.py in JSON_to_array(split_perc, file)
     12 
     13     ice_berg=[np.array(i) for i in train_set['is_iceberg']]
---> 14     ice_berg = np.array(ice_berg).reshape(1, 1604)
     15     inc_set_ice_berg = [np.array(i) for i in inc_set['is_iceberg']]
     16     inc_set_ice_berg = np.array(inc_set_ice_berg).reshape(1, 1471)

ValueError: cannot reshape array of size 1283 into shape (1,1604)

## === cell 74
X_train_band_1, X_val_test_band_1, y_train_band_1, y_val_test_band_1 = band_1_array
d_band_1 = model(X_train_band_1, y_train_band_1, X_val_test_band_1, y_val_test_band_1, num_iterations = 15000, learning_rate = 0.005, print_cost = True)


## --- ERROR in cell 74, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4128960092.py in <cell line: 0>()
----> 1 X_train_band_1, X_val_test_band_1, y_train_band_1, y_val_test_band_1 = band_1_array
      2 d_band_1 = model(X_train_band_1, y_train_band_1, X_val_test_band_1, y_val_test_band_1, num_iterations = 15000, learning_rate = 0.005, print_cost = True)

NameError: name 'band_1_array' is not defined

## === cell 77
X_train_band_2, X_val_test_band_2, y_train_band_2, y_val_test_band_2 = band_2_array
d_band_2 = model(X_train_band_2, y_train_band_2, X_val_test_band_2, y_val_test_band_2, num_iterations = 15000, learning_rate = 0.001, print_cost = True)


## --- ERROR in cell 77, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2125833770.py in <cell line: 0>()
----> 1 X_train_band_2, X_val_test_band_2, y_train_band_2, y_val_test_band_2 = band_2_array
      2 d_band_2 = model(X_train_band_2, y_train_band_2, X_val_test_band_2, y_val_test_band_2, num_iterations = 15000, learning_rate = 0.001, print_cost = True)

NameError: name 'band_2_array' is not defined

## === cell 79
X_train_inc, X_val_test_inc, y_train_inc, y_val_test_inc = inc_ang_array
d_inc_ang = model(X_train_inc, y_train_inc, X_val_test_inc, y_val_test_inc, num_iterations = 15000, learning_rate = 0.001, print_cost = True)


## --- ERROR in cell 79, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2225898087.py in <cell line: 0>()
----> 1 X_train_inc, X_val_test_inc, y_train_inc, y_val_test_inc = inc_ang_array
      2 d_inc_ang = model(X_train_inc, y_train_inc, X_val_test_inc, y_val_test_inc, num_iterations = 15000, learning_rate = 0.001, print_cost = True)

NameError: name 'inc_ang_array' is not defined

## === cell 81
test_band_1=[np.array(i) for i in test['band_1']]
test_band_2=[np.array(i) for i in test['band_2']]
test_inc_ang=[np.array(i) for i in test['inc_angle']]

max_band_1 = np.max(np.array(np.abs(test_band_1)))
max_band_2 = np.max(np.array(np.abs(test_band_2)))
max_inc_ang = np.max(np.array(np.abs(test_inc_ang)))

band_2 = np.array(test_band_2).T/max_band_2
band_1 = np.array(test_band_1).T/max_band_1
inc_ang = (np.array(test_inc_ang).T/max_inc_ang).reshape(1, 8424)

binary_result_band_1, band_1_Y_prediction_test = predict(d_band_1['w'], d_band_1['b'], band_1)
y_pred_band_1 = pd.DataFrame(band_1_Y_prediction_test.T,columns=['is_iceberg'])

binary_result_band_2, band_2_Y_prediction_test = predict(d_band_2['w'], d_band_2['b'], band_2)
y_pred_band_2 = pd.DataFrame(band_2_Y_prediction_test.T,columns=['is_iceberg'])

binary_result_band_3, inc_ang_Y_prediction_test = predict(d_inc_ang['w'], d_inc_ang['b'], inc_ang)
y_pred_inc_ang = pd.DataFrame(inc_ang_Y_prediction_test.T,columns=['is_iceberg'])




## --- ERROR in cell 81, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1320775633.py in <cell line: 0>()
      5 max_band_1 = np.max(np.array(np.abs(test_band_1)))
      6 max_band_2 = np.max(np.array(np.abs(test_band_2)))
----> 7 max_inc_ang = np.max(np.array(np.abs(test_inc_ang)))
      8 
      9 band_2 = np.array(test_band_2).T/max_band_2

UFuncTypeError: ufunc 'absolute' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> None

## === cell 84
y_pred = pd.DataFrame()
y_pred['id'] = test['id']
y_pred['is_iceberg'] = ((y_pred_band_1['is_iceberg']) + (y_pred_band_2['is_iceberg']) +(y_pred_inc_ang['is_iceberg']))/3
y_pred.to_csv('submission_3.csv', index=False)


## --- ERROR in cell 84, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3497474579.py in <cell line: 0>()
      2 y_pred['id'] = test['id']
      3 #y_pred['is_iceberg'] = None
----> 4 y_pred['is_iceberg'] = ((y_pred_band_1['is_iceberg']) + (y_pred_band_2['is_iceberg']) +(y_pred_inc_ang['is_iceberg']))/3
      5 y_pred.to_csv('submission_3.csv', index=False)

NameError: name 'y_pred_band_1' is not defined

## === cell 91
y_pred.head()


## === cell 102
y_pred


## === cell 125
y_pred


## === cell 127
len(test)


## === cell 128
test['prediction'] = None


## === cell 129
test['prediction'] = 


## --- ERROR in cell 129, traceback:
  File "/tmp/ipykernel_11/415101739.py", line 1
    test['prediction'] =
                         ^
SyntaxError: invalid syntax


## === cell 136
d['w'].shape


## --- ERROR in cell 136, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3974249156.py in <cell line: 0>()
----> 1 d['w'].shape

NameError: name 'd' is not defined

## === cell 140
band1_x_train, band1_x_train 


## --- ERROR in cell 140, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3769711853.py in <cell line: 0>()
----> 1 band1_x_train, band1_x_train

NameError: name 'band1_x_train' is not defined
