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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.54983

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


## === cell 1
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import LearningRateScheduler
import xgboost as xg
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")


## === cell 3
train_df.head()


## === cell 4
test_df.head()


## === cell 5
print(train_df.shape)
print(test_df.shape)


## === cell 6
X = np.array(train_df[train_df.columns[1:-1]])
y = np.array(train_df['Pawpularity'])


## === cell 7
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.15,random_state=244)


## === cell 8
model1 = RandomForestRegressor(max_depth = 15)

model1.fit(xtr,ytr)

predictions = model1.predict(xte)

print("Error: " , np.sqrt(mean_squared_error(yte, predictions)))


## === cell 9
model2 = xg.XGBRegressor(n_estimators = 512,max_depth = 20,objective ='reg:squarederror')
model2.fit(xtr, ytr)

predictions = model2.predict(xte)

print("Error: " , np.sqrt(mean_squared_error(yte, predictions)))


## === cell 10
poly2 = PolynomialFeatures(degree=2)
poly3 = PolynomialFeatures(degree=3)
poly4 = PolynomialFeatures(degree=4)
poly5 = PolynomialFeatures(degree=5)
poly6 = PolynomialFeatures(degree=6)
poly7 = PolynomialFeatures(degree=7)

xtr2 = poly2.fit_transform(xtr)
xtr3 = poly3.fit_transform(xtr)
xtr4 = poly4.fit_transform(xtr)
xtr5 = poly5.fit_transform(xtr)
xtr6 = poly6.fit_transform(xtr)
xtr7 = poly7.fit_transform(xtr)


model3 = LinearRegression()

model3.fit(xtr2,ytr)
predictions = model3.predict(poly2.fit_transform(xte))
print("Polynomial Regression error(degree=2): ", np.sqrt(mean_squared_error(yte,predictions)))
model3.fit(xtr3,ytr)
predictions = model3.predict(poly3.fit_transform(xte))
print("Polynomial Regression error(degree=3): ", np.sqrt(mean_squared_error(yte,predictions)))
model3.fit(xtr4,ytr)
predictions = model3.predict(poly4.fit_transform(xte))
print("Polynomial Regression error(degree=4): ", np.sqrt(mean_squared_error(yte,predictions)))


## === cell 11
X2 = poly2.fit_transform(X)

model3.fit(X2,y)
predictions = model3.predict(poly2.fit_transform(X))
print("Polynomial Regression error(degree=2): ", np.sqrt(mean_squared_error(y,predictions)))


## === cell 12
model4 = Sequential()
model4.add(InputLayer(input_shape=(xtr.shape[1],)))

for _ in range(3):
    model4.add(Dense(128,activation="relu",kernel_initializer="normal"))

model4.add(Dense(1,activation="linear",kernel_initializer="normal"))

optim = Adam(
    learning_rate=0.0005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

model4.compile(loss="mse",optimizer='adam',metrics="mse")
model4.summary()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3281615193.py in <cell line: 0>()
     16 )
     17 
---> 18 model4.compile(loss="mse",optimizer='adam',metrics="mse")
     19 model4.summary()

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in __init__(self, metrics, weighted_metrics, name, output_names)
    132         super().__init__(name=name)
    133         if metrics and not isinstance(metrics, (list, tuple, dict)):
--> 134             raise ValueError(
    135                 "Expected `metrics` argument to be a list, tuple, or dict. "
    136                 f"Received instead: metrics={metrics} of type {type(metrics)}"

ValueError: Expected `metrics` argument to be a list, tuple, or dict. Received instead: metrics=mse of type <class 'str'>

## === cell 13
model4.fit(xtr,ytr,batch_size=32,epochs=100,validation_data=(xte,yte))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1272489889.py in <cell line: 0>()
----> 1 model4.fit(xtr,ytr,batch_size=32,epochs=100,validation_data=(xte,yte))

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 14
print("Error on test data: " , np.sqrt(model4.evaluate(xte, yte,verbose = 0)[1]))
print("Error on whole data: " ,np.sqrt(model4.evaluate(X, y,verbose = 0)[1]))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3841054061.py in <cell line: 0>()
----> 1 print("Error on test data: " , np.sqrt(model4.evaluate(xte, yte,verbose = 0)[1]))
      2 print("Error on whole data: " ,np.sqrt(model4.evaluate(X, y,verbose = 0)[1]))

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 15
submission_x = np.array(test_df[test_df.columns[1:]])
Id = np.array(test_df['Id'])


## === cell 16
predictions = model4.predict(submission_x).reshape(submission_x.shape[0],)
submission_df = pd.DataFrame()

submission_df['Id'] = Id
submission_df['Pawpularity'] = predictions
submission_df.to_csv('submission.csv',index=False)


## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
