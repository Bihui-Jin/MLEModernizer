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

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from abc import ABC, abstractmethod
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.random.set_seed(1234)
np.random.seed(1234)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def resolve_input_base() -> str:
    candidates = [
        "/kaggle/input/petfinder-pawpularity-score/",
        "../input/petfinder-pawpularity-score/",
        "/kaggle/data/petfinder-pawpularity-score/",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p if p.endswith("/") else p + "/"
    return "../input/petfinder-pawpularity-score/"


INPUT_BASE = resolve_input_base()
print("Using INPUT_BASE:", INPUT_BASE)




## === cell 2
class Pipeline(ABC):
    input_path = INPUT_BASE


class ImagePipeline(Pipeline):
    X: tf.keras.preprocessing.image.DirectoryIterator

    def get_dataframe(self, csv: str):
        self.df = pd.read_csv(self.input_path + csv)[["Id"]]
        self.df.index = self.df.Id.astype(str) + ".jpg"
        self.df.drop("Id", axis=1, inplace=True)

    def __init__(
        self,
        csv: str,
        datagen: Optional[ImageDataGenerator] = None,
        seed: Optional[int] = 1234,
    ):
        self.datagen = datagen or ImageDataGenerator(rescale=1.0 / 255.0)
        self.get_dataframe(csv)

        img_dir = self.input_path + csv.replace(".csv", "")
        self.X = self.datagen.flow_from_dataframe(
            dataframe=self.df.reset_index().rename(columns={"index": "Id"}),
            directory=img_dir,
            x_col="Id",
            class_mode=None,
            target_size=(224, 224),
            batch_size=32,
            shuffle=False,
            seed=seed,
        )


class POCEncoder:
    def __init__(self):
        candidate_paths = [
            "../input/01-keras-images-model-poc/poc.h5",
            "/kaggle/input/01-keras-images-model-poc/poc.h5",
        ]
        model_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError(
                "POC model file not found in expected input dataset: poc.h5"
            )

        poc = tf.keras.models.load_model(model_path, compile=False)
        self.poc_extract = tf.keras.models.Model(
            poc.input, poc.get_layer("encode").output
        )

    def predict(self, X):
        return self.poc_extract.predict(X, verbose=0)


class ImportPipeline(ABC):
    @abstractmethod
    def import_data(self):
        pass


class ImportPOCPipeline(ImportPipeline):
    def __init__(self):
        self.poc_encoder = POCEncoder()

    def import_data(self):
        candidate_paths = [
            "../input/02-encode/train_poc_encoded.csv",
            "/kaggle/input/02-encode/train_poc_encoded.csv",
        ]
        train_enc_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                train_enc_path = p
                break
        if train_enc_path is None:
            raise FileNotFoundError(
                "Encoded train features not found in expected input dataset: train_poc_encoded.csv"
            )

        self.train_encode = pd.read_csv(train_enc_path).set_index("Id")
        self.selected_columns = self.train_encode.columns

        self.test_image_pipeline = ImagePipeline("test.csv")
        self.test_image_encoded = self.poc_encoder.predict(self.test_image_pipeline.X)

        self.test_encode = pd.DataFrame(self.test_image_encoded)
        self.test_encode.index = self.test_image_pipeline.df.index.str.replace(
            ".jpg", "", regex=False
        )
        self.test_encode.columns = "poc_" + self.test_encode.columns.astype(str)

        self.test_encode = self.test_encode[self.selected_columns]




## === cell 3
class MainPipeline(Pipeline):
    mergepocpipeline = ImportPOCPipeline()
    mergepocpipeline.import_data()


class TrainPipeline(MainPipeline):
    def __init__(self, filepath, target):
        self.df = pd.read_csv(filepath).set_index("Id")
        self.target = target

    def run(self):
        self.y = self.df[self.target]
        self.X = self.df.drop(self.target, axis=1)
        self.X = pd.concat([self.X, self.mergepocpipeline.train_encode], axis=1)


class TestPipeline(MainPipeline):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath).set_index("Id")

    def run(self):
        self.X = self.df
        self.X = pd.concat([self.X, self.mergepocpipeline.test_encode], axis=1)


train = TrainPipeline(INPUT_BASE + "train.csv", target="Pawpularity")
train.run()

test = TestPipeline(INPUT_BASE + "test.csv")
test.run()

print("Train shape:", train.X.shape, "Test shape:", test.X.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/72249696.py in <cell line: 0>()
      1 # Fix: ensure merge pipeline runs before Train/Test pipelines.
----> 2 class MainPipeline(Pipeline):
      3     mergepocpipeline = ImportPOCPipeline()
      4     mergepocpipeline.import_data()
      5 

/tmp/ipykernel_55/72249696.py in MainPipeline()
      1 # Fix: ensure merge pipeline runs before Train/Test pipelines.
      2 class MainPipeline(Pipeline):
----> 3     mergepocpipeline = ImportPOCPipeline()
      4     mergepocpipeline.import_data()
      5 

/tmp/ipykernel_55/45828289.py in __init__(self)
     69 class ImportPOCPipeline(ImportPipeline):
     70     def __init__(self):
---> 71         self.poc_encoder = POCEncoder()
     72 
     73     def import_data(self):

/tmp/ipykernel_55/45828289.py in __init__(self)
     48                 break
     49         if model_path is None:
---> 50             raise FileNotFoundError(
     51                 "POC model file not found in expected input dataset: poc.h5"
     52             )

FileNotFoundError: POC model file not found in expected input dataset: poc.h5

## === cell 4
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


try:
    from catboost import CatBoostRegressor  # type: ignore

    def fit_catboost(X_train, y_train, X_test, y_test):
        reg = CatBoostRegressor(
            iterations=5000,
            grow_policy="Lossguide",
            loss_function="RMSE",
            verbose=False,
            random_seed=1234,
        )
        reg.fit(X_train, y_train)
        return reg

    fit_models = [fit_catboost]
except Exception as e:
    print(
        "CatBoost unavailable; falling back to HistGradientBoostingRegressor. Reason:",
        repr(e),
    )
    from sklearn.ensemble import HistGradientBoostingRegressor

    def fit_catboost(X_train, y_train, X_test, y_test):
        reg = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=None,
            random_state=1234,
        )
        reg.fit(X_train, y_train)
        return reg

    fit_models = [fit_catboost]

kf = KFold(n_splits=5, shuffle=True, random_state=1234)



## === cell 5
fit_results = {}
for k, (train_index, test_index) in enumerate(kf.split(train.X)):
    print(f"K Fold: {k + 1}")
    X_train, X_valid = train.X.iloc[train_index], train.X.iloc[test_index]
    y_train, y_valid = train.y.iloc[train_index], train.y.iloc[test_index]

    for fit_model in fit_models:
        model_name = (
            "_".join(fit_model.__name__.split("fit_")[1:]) or fit_model.__name__
        )
        model_result_path = f"model_results/{model_name}"
        model_checkpoint = f"{model_result_path}/fold_{k+1}.pickle"

        os.makedirs(model_result_path, exist_ok=True)
        if model_name not in fit_results:
            fit_results[model_name] = []

        if os.path.isfile(model_checkpoint):
            with open(model_checkpoint, "rb") as f:
                model = pickle.load(f)
        else:
            model = fit_model(X_train, y_train, X_valid, y_valid)
            with open(model_checkpoint, "wb") as f:
                pickle.dump(model, f)

        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)

        rmse_train = rmse(y_train, train_pred)
        rmse_valid = rmse(y_valid, valid_pred)

        print(f"rmse train: {rmse_train:.6f}")
        print(f"rmse valid: {rmse_valid:.6f}")

        fit_results[model_name].append(
            {"model": model, "rmse_train": rmse_train, "rmse_test": rmse_valid}
        )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2323936995.py in <cell line: 0>()
      1 # Fix: ensure output directory exists and the CV loop runs end-to-end.
      2 fit_results = {}
----> 3 for k, (train_index, test_index) in enumerate(kf.split(train.X)):
      4     print(f"K Fold: {k + 1}")
      5     X_train, X_valid = train.X.iloc[train_index], train.X.iloc[test_index]

NameError: name 'train' is not defined

## === cell 6
model_name = list(fit_results.keys())[0]
cv_mean = float(np.mean([score["rmse_test"] for score in fit_results[model_name]]))
cv_std = float(np.std([score["rmse_test"] for score in fit_results[model_name]]))
print("Model:", model_name, "CV mean RMSE:", cv_mean, "CV std:", cv_std)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3925601017.py in <cell line: 0>()
      1 # Fix: summary stats cells referenced undefined variables previously.
----> 2 model_name = list(fit_results.keys())[0]
      3 cv_mean = float(np.mean([score["rmse_test"] for score in fit_results[model_name]]))
      4 cv_std = float(np.std([score["rmse_test"] for score in fit_results[model_name]]))
      5 print("Model:", model_name, "CV mean RMSE:", cv_mean, "CV std:", cv_std)

IndexError: list index out of range

## === cell 7
sub_pred = np.mean(
    [m["model"].predict(test.X) for m in fit_results[model_name]],
    axis=0,
)

sub_pred = np.clip(sub_pred, 0.0, 100.0)

sub = pd.DataFrame({"Id": test.X.index.astype(str), "Pawpularity": sub_pred})
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2863830929.py in <cell line: 0>()
      1 # Predict on test by averaging fold models (same original logic).
      2 sub_pred = np.mean(
----> 3     [m["model"].predict(test.X) for m in fit_results[model_name]],
      4     axis=0,
      5 )

NameError: name 'model_name' is not defined
