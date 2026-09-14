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

3.10

# 3. Installed packages

No external packages required in the script and installed.

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

20.979486386764705

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from abc import ABC, abstractmethod

from typing import Optional, Callable
import numpy as np

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
class Pipeline(ABC):
    input_path = '../input/petfinder-pawpularity-score/'


class ImagePipeline(Pipeline):
    X: tf.keras.preprocessing.image.DirectoryIterator
    
    def get_dataframe(self, csv: str):
        self.df = pd.read_csv(
            self.input_path + csv,
        )[['Id']]
        self.df.index = self.df.Id
        self.df.index = self.df.index.astype(str) + ".jpg"
        self.df.drop("Id", axis=1, inplace=True)
        
    def __init__(
        self,
        csv: str,
        datagen: Optional[ImageDataGenerator]=None,
        seed: Optional[int]=1234
    ):
        self.datagen = ImageDataGenerator(
            rescale=1./255
        )
        self.get_dataframe(csv)
        self.X = self.datagen.flow_from_dataframe(
            dataframe=self.df.reset_index(),
            directory=self.input_path + csv.replace(".csv", ""),
            x_col="Id",
            class_mode=None,
            target_size=(224, 224),
            batch_size=32,
            shuffle=False,
            seed=seed
        )

class POCEncoder:
    def __init__(self):
        poc = tf.keras.models.load_model("../input/01-keras-images-model-poc/poc.h5")
        poc.summary()
        self.poc_extract = tf.keras.models.Model(poc.input, poc.get_layer("encode").output)
    def predict(self, X):
        return self.poc_extract.predict(X)
    

class ImportPipeline(ABC):
    @abstractmethod
    def import_data(self):
        pass
    
    
class ImportPOCPipeline(ImportPipeline):
    def __init__(self):
        self.poc_encoder = POCEncoder()
        
    def import_data(self):
        self.train_encode = pd.read_csv("../input/02-encode/train_poc_encoded.csv")
        self.train_encode = self.train_encode.set_index("Id")
        self.selected_columns = self.train_encode.columns

        self.test_image_pipeline = ImagePipeline("test.csv")
        self.test_image_encoded = self.poc_encoder.predict(self.test_image_pipeline.X)

        self.test_encode = pd.DataFrame(self.test_image_encoded)
        self.test_encode.index = self.test_image_pipeline.df.index
        self.test_encode.columns =  "poc_"+ self.test_encode.columns.astype(str)

        self.test_encode = pd.concat([self.test_image_pipeline.df, self.test_encode], axis=1)
        self.test_encode.index = self.test_encode.index.str.replace(".jpg", "")
        self.test_encode = self.test_encode[self.selected_columns]

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2014860559.py in <cell line: 0>()
----> 1 class Pipeline(ABC):
      2     input_path = '../input/petfinder-pawpularity-score/'
      3 
      4 
      5 class ImagePipeline(Pipeline):

NameError: name 'ABC' is not defined

## === cell 6
class MainPipeline(Pipeline):
    mergepocpipeline = ImportPOCPipeline()
    mergepocpipeline.import_data()

class TrainPipeline(MainPipeline):
    def __init__(self, filepath, target):
        self.df = pd.read_csv(filepath)
        self.df = self.df.set_index("Id")
        self.target = target
    def run(self):
        self.y = self.df[self.target]
        self.X = self.df.drop(self.target, axis=1)
        self.X = pd.concat([self.X, self.mergepocpipeline.train_encode], axis=1)

class TestPipeline(MainPipeline):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = self.df.set_index("Id")
    def run(self):
        self.X = self.df
        self.X = pd.concat([self.X, self.mergepocpipeline.test_encode], axis=1)
        
        
train = TrainPipeline("../input/petfinder-pawpularity-score/train.csv", target='Pawpularity')
train.run()

test = TestPipeline("../input/petfinder-pawpularity-score/test.csv")
test.run()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3525148095.py in <cell line: 0>()
----> 1 class MainPipeline(Pipeline):
      2     mergepocpipeline = ImportPOCPipeline()
      3     mergepocpipeline.import_data()
      4 
      5 class TrainPipeline(MainPipeline):

NameError: name 'Pipeline' is not defined

## === cell 8
import os
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
import pickle
import json
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error



def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5


def fit_catboost(X_train, y_train, X_test, y_test):
    reg = CatBoostRegressor(
        iterations=5000,
        grow_policy='Lossguide',
        loss_function='RMSE',
        verbose=False,
    )
    reg.fit(X_train, y_train)
    return reg


fit_models = [fit_catboost]
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

fit_results = {}
for k, (train_index, test_index) in enumerate(kf.split(train.X)):
    print(f"K Fold: {k + 1}")
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train.X.iloc[train_index], train.X.iloc[test_index]
    y_train, y_test = train.y.iloc[train_index], train.y.iloc[test_index]
    
    for fit_model in fit_models:
        model_name = '_'.join(fit_model.__name__.split('fit_')[1:])
        model_result_path = f"model_results/{model_name}"
        model_checkpoint = f"model_results/{model_name}/fold_{k+1}.pickle"
        
        if not os.path.exists(model_result_path):
            os.makedirs(model_result_path)
        if model_name not in fit_results:
            fit_results[model_name] = []
        if os.path.isfile(model_checkpoint):
            print("Found trained model")
            with open(model_checkpoint, 'rb') as f:
                model = pickle.load(f)

        else:
            model = fit_model(X_train, y_train, X_test, y_test)
        
        train_pred = model.predict(X_train)
        rmse_train = rmse(y_train, train_pred)

        test_pred = model.predict(X_test)
        rmse_test = rmse(y_test, test_pred)
        
        print(f"rmse train: {rmse_train}")
        print(f"rmse test: {rmse_test}")
        
        try:
            oof_pred = model.predict(X_oof)
            rmse_oof = rmse(y_oof, oof_pred)
            print(f"rmse OOF: {rmse_oof}")
        except NameError:
            pass

        try:
            with open(f"model_results/{model_name}/fold_{k+1}.pickle", "wb") as f:
                pickle.dump(model, f)
        except:
            pass
        fit_results[model_name].append({
            'model': model,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test
        })

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2926297716.py in <cell line: 0>()
     29 
     30 fit_results = {}
---> 31 for k, (train_index, test_index) in enumerate(kf.split(train.X)):
     32     print(f"K Fold: {k + 1}")
     33     print("TRAIN:", train_index, "TEST:", test_index)

NameError: name 'train' is not defined

## === cell 9
np.mean([score['rmse_test'] for score in fit_results[model_name]])

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1849690116.py in <cell line: 0>()
----> 1 np.mean([score['rmse_test'] for score in fit_results[model_name]])

NameError: name 'np' is not defined

## === cell 10
np.std([score['rmse_test'] for score in fit_results[model_name]])

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2369524173.py in <cell line: 0>()
----> 1 np.std([score['rmse_test'] for score in fit_results[model_name]])

NameError: name 'np' is not defined

## === cell 11
sub = np.mean(
    [model['model'].predict(test.X) for model in fit_results[model_name]],
    axis=0
)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2622760010.py in <cell line: 0>()
----> 1 sub = np.mean(
      2     [model['model'].predict(test.X) for model in fit_results[model_name]],
      3     axis=0
      4 )

NameError: name 'np' is not defined

## === cell 12
sub = pd.DataFrame(sub)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3642909098.py in <cell line: 0>()
----> 1 sub = pd.DataFrame(sub)

NameError: name 'pd' is not defined

## === cell 13
sub.columns = ['Pawpularity']
sub.index = test.X.index
sub.to_csv("submission.csv")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/687083531.py in <cell line: 0>()
----> 1 sub.columns = ['Pawpularity']
      2 sub.index = test.X.index
      3 sub.to_csv("submission.csv")

NameError: name 'sub' is not defined
