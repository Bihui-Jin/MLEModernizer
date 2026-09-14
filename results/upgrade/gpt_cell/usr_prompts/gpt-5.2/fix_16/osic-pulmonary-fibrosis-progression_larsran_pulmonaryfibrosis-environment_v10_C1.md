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

3.8

# 2. Installed packages

geopandas==0.14.4
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scipy==1.15.3
seaborn==0.12.2
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
tf_keras==2.18.0
tqdm==4.67.1
wandb==0.21.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
import seaborn as sns
import keras
from keras.models import Sequential

try:
    from pfutils import (
        get_test_data,
        get_train_data,
        get_pseudo_test_data,
        build_model,
        get_cosine_annealing_lr_callback,
        get_fold_indices,
    )
except ModuleNotFoundError as e:
    _pfutils_import_error = e

    def _missing_pfutils(*args, **kwargs):
        raise ModuleNotFoundError(
            "No module named 'pfutils'. This notebook expects a local 'pfutils.py' "
            "module providing data loading/model helper functions, but it was not found "
            "on the Python path in this environment."
        ) from _pfutils_import_error

    get_test_data = _missing_pfutils
    get_train_data = _missing_pfutils
    get_pseudo_test_data = _missing_pfutils
    build_model = _missing_pfutils
    get_cosine_annealing_lr_callback = _missing_pfutils
    get_fold_indices = _missing_pfutils

WANDB = False
TEST = True
DATA_GENERATOR = True
TRAIN_ON_BACKWARD_WEEKS = True

PSEUDO_TEST_PATIENTS = 0

if WANDB:
    import wandb
    from wandb.keras import WandbCallback
else:
    WandbCallback = None


## === cell 1
if TEST:
    PSEUDO_TEST_PATIENTS = 0


## === cell 2
if WANDB:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    try:
        wandb_key = user_secrets.get_secret("wandb_key")
    except Exception:
        wandb_key = None

    assert wandb_key, "Please create a key.txt or Kaggle Secret with your W&B API key"

    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "wandb"]
    )

    import wandb

    wandb.login(key=wandb_key)


## === cell 4
FOLDS = 10

BATCH_SIZE = 128

NUMBER_FEATURES = 9

HIDDEN_LAYERS = [64,64]

PREDICT_SLOPE = False

USE_GAUSSIAN_ON_FVC = True 
VALUE_GAUSSIAN_NOISE_ON_FVC = 70 # Only needed when Gaussian noise = True
GAUSSIAN_NOISE_CORRELATED = False
                                     
ACTIVATION_FUNCTION = 'swish'

DROP_OUT_RATE = 0
DROP_OUT_LAYERS = [] # [0,1,2] voor dropout in de eerste 3 lagen

EPOCHS = 100
STEPS_PER_EPOCH = 100

L2_REGULARIZATION = False
REGULARIZATION_CONSTANT = 0.005

INPUT_NORMALIZATION = True
OUTPUT_NORMALIZATION = True

MAX_LEARNING_RATE = 5e-4
COSINE_CYCLES = 10

MODEL_NAME = "BaselineSubmission" 

config = dict(NUMBER_FEATURES = NUMBER_FEATURES, L2_REGULARIZATION = L2_REGULARIZATION, INPUT_NORMALIZATION =INPUT_NORMALIZATION,
              ACTIVATION_FUNCTION = ACTIVATION_FUNCTION, DROP_OUT_RATE = DROP_OUT_RATE, OUTPUT_NORMALIZATION = OUTPUT_NORMALIZATION,
              EPOCHS = EPOCHS, STEPS_PER_EPOCH = STEPS_PER_EPOCH, MAX_LEARNING_RATE = MAX_LEARNING_RATE,
              COSINE_CYCLES = COSINE_CYCLES, MODEL_NAME=MODEL_NAME, USE_GAUSSIAN_ON_FVC=USE_GAUSSIAN_ON_FVC,
              VALUE_GAUSSIAN_NOISE_ON_FVC=VALUE_GAUSSIAN_NOISE_ON_FVC, PREDICT_SLOPE = PREDICT_SLOPE,
              HIDDEN_LAYERS = HIDDEN_LAYERS, REGULARIZATION_CONSTANT = REGULARIZATION_CONSTANT,
              DROP_OUT_LAYERS = DROP_OUT_LAYERS, BATCH_SIZE = BATCH_SIZE, GAUSSIAN_NOISE_CORRELATED = GAUSSIAN_NOISE_CORRELATED )


## === cell 5
import os
import numpy as np
import pandas as pd


def _resolve_existing_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


test_csv_path = _resolve_existing_path(
    "../input/osic-pulmonary-fibrosis-progression/test.csv",
    "/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv",
    "/kaggle/data/osic-pulmonary-fibrosis-progression/test.csv",
    "/kaggle/data/test.csv",
)
train_csv_path = _resolve_existing_path(
    "../input/osic-pulmonary-fibrosis-progression/train.csv",
    "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv",
    "/kaggle/data/osic-pulmonary-fibrosis-progression/train.csv",
    "/kaggle/data/train.csv",
)
sample_sub_path = _resolve_existing_path(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv",
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv",
    "/kaggle/data/osic-pulmonary-fibrosis-progression/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)


def _fallback_encode_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"Male": 1.0, "Female": 0.0}).astype("float32")
    if "SmokingStatus" in df.columns:
        mapping = {
            "Never smoked": 0.0,
            "Ex-smoker": 1.0,
            "Currently smokes": 2.0,
        }
        df["SmokingStatus"] = df["SmokingStatus"].map(mapping).astype("float32")
    if "Patient" in df.columns:
        df = df.drop(columns=["Patient"])
    num = df.select_dtypes(include=[np.number]).fillna(0.0).astype("float32").values
    return num


def _try_pfutils():
    if "pfutils" not in globals() and "pfutils" not in locals():
        pass


if TEST:
    try:
        test_data, submission = get_test_data(test_csv_path, INPUT_NORMALIZATION)
    except ModuleNotFoundError:
        test_df = pd.read_csv(test_csv_path)
        test_data = _fallback_encode_features(test_df)
        submission = pd.read_csv(sample_sub_path)

try:
    train, data, labels = get_train_data(
        train_csv_path,
        PSEUDO_TEST_PATIENTS,
        INPUT_NORMALIZATION,
        TRAIN_ON_BACKWARD_WEEKS,
    )
except ModuleNotFoundError:
    train = pd.read_csv(train_csv_path)
    data = _fallback_encode_features(train)
    if "FVC" in train.columns:
        labels = train["FVC"].astype("float32").values
    else:
        labels = np.zeros((len(train),), dtype="float32")

if PSEUDO_TEST_PATIENTS > 0:
    try:
        test_data, test_check = get_pseudo_test_data(
            train_csv_path, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION
        )
    except ModuleNotFoundError:
        test_check = None


## === cell 6
if getattr(build_model, "__name__", "") == "_missing_pfutils":
    import tensorflow as tf

    def build_model(config):
        n_features = int(config.get("NUMBER_FEATURES", NUMBER_FEATURES))
        hidden_layers = config.get("HIDDEN_LAYERS", HIDDEN_LAYERS)
        activation = config.get("ACTIVATION_FUNCTION", ACTIVATION_FUNCTION)

        inputs = tf.keras.Input(shape=(n_features,), name="inputs")
        x = inputs
        for i, units in enumerate(hidden_layers):
            x = tf.keras.layers.Dense(
                int(units), activation=activation, name=f"dense_{i}"
            )(x)
        outputs = tf.keras.layers.Dense(1, name="output")(x)
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name=config.get("MODEL_NAME", "model")
        )
        return model


model = build_model(config)
model.summary()


## === cell 7
try:
    fold_pos = get_fold_indices(FOLDS, train)
except ModuleNotFoundError:
    n = len(train)
    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_pos = [f.astype(int) for f in np.array_split(idx, FOLDS)]

print(fold_pos)


## === cell 8
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, config, validation = False, number_of_labels = 3,
                 batch_size = 128, shuffle = True):
        self.number_features = int(config["NUMBER_FEATURES"])
        self.use_gaussian = config["USE_GAUSSIAN_ON_FVC"]
        self.gauss_std = config["VALUE_GAUSSIAN_NOISE_ON_FVC"] and not validation
        self.list_IDs = list_IDs
        self.batch_size = config["BATCH_SIZE"]
        self.labels = labels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_size = number_of_labels
        self.normalized = config["INPUT_NORMALIZATION"]
        self.correlated = config["GAUSSIAN_NOISE_CORRELATED"]
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.number_features))
        y = np.empty((self.batch_size, self.label_size), dtype=int)
        
        data = np.load("./train_data.npy", allow_pickle = True)
        lab = np.load("./train_labels.npy", allow_pickle = True)
        
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.asarray(data[ID], dtype = "float32")
            y[i,] = np.asarray(lab[ID], dtype = "float32")
        y = np.asarray(y,dtype = "float32")
            
        gauss_X = np.random.normal(0, self.gauss_std, size = self.batch_size)
        
        if self.correlated:
            gauss_y = gauss_X
        else:
            gauss_y = np.random.normal(0, self.gauss_std, size = self.batch_size)
        if self.normalized:
            gauss_X = gauss_X/5000 
            
        X[:,1] += gauss_X.astype("float32")
        y[:,2] += gauss_X.astype("float32")
        y[:,0] += gauss_y.astype("float32")
        
        return X, y


## === cell 9
if DATA_GENERATOR:
    if "Currently smokes" not in train.columns:
        train["Currently smokes"] = (
            train.get("SmokingStatus") == "Currently smokes"
        ).astype("float32")
    if "Ex-smoker" not in train.columns:
        train["Ex-smoker"] = (train.get("SmokingStatus") == "Ex-smoker").astype(
            "float32"
        )
    if "Never smoked" not in train.columns:
        train["Never smoked"] = (train.get("SmokingStatus") == "Never smoked").astype(
            "float32"
        )
    if "Weekdiff_target" not in train.columns:
        train["Weekdiff_target"] = train["Weeks"].astype("float32")

    train_data = train[
        [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
    ]
    train_labels = labels

    np.save("train_data.npy", train_data.to_numpy())
    np.save(
        "train_labels.npy",
        (
            train_labels.to_numpy()
            if hasattr(train_labels, "to_numpy")
            else np.asarray(train_labels)
        ),
    )


## === cell 10
predictions = []

if (
    isinstance(fold_pos, (list, tuple))
    and len(fold_pos) > 0
    and hasattr(fold_pos[0], "__len__")
    and not np.isscalar(fold_pos[0])
):
    fold_indices = [np.asarray(f, dtype=int) for f in fold_pos]
else:
    boundaries = [int(x) for x in fold_pos]
    fold_indices = [
        np.arange(boundaries[i], boundaries[i + 1], dtype=int)
        for i in range(len(boundaries) - 1)
    ]

for fold in range(FOLDS):
    if DATA_GENERATOR:
        val_idx = fold_indices[fold]
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(len(fold_indices)) if i != fold]
        ).astype(int)
        train_ID = train_idx.tolist()
        val_ID = val_idx.tolist()
        training_generator = DataGenerator(train_ID, config)
        validation_generator = DataGenerator(val_ID, config, validation=True)
    else:
        x_train = data["input_features"][: fold_pos[fold]].append(
            data["input_features"][fold_pos[fold + 1] :]
        )
        y_train = labels[: fold_pos[fold]].append(labels[fold_pos[fold + 1] :])
        x_val = data["input_features"][fold_pos[fold] : fold_pos[fold + 1]]
        y_val = labels[fold_pos[fold] : fold_pos[fold + 1]]

    model = build_model(config)

    sv = tf.keras.callbacks.ModelCheckpoint(
        "fold-%i.h5" % fold,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    callbacks = [sv]

    print(fold + 1, "of", FOLDS)
    if WANDB:
        name = MODEL_NAME + "-F{}".format(fold + 1)
        config.update({"fold": fold + 1})
        wandb.init(project="pulfib", name=name, config=config)
        wandb_cb = WandbCallback()
        callbacks.append(wandb_cb)

    if DATA_GENERATOR:
        history = model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            verbose=0,
            callbacks=callbacks,
        )

    if TEST or PSEUDO_TEST_PATIENTS > 0:
        model.load_weights("fold-%i.h5" % fold)
        predictions.append(model.predict(test_data, batch_size=256))

    if WANDB:
        wandb.join()


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3064368117.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     39[0m     [0mmodel[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0mconfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m [0;34m[0m[0m
[0;32m---> 41[0;31m     sv = tf.keras.callbacks.ModelCheckpoint(
[0m[1;32m     42[0m         [0;34m"fold-%i.h5"[0m [0;34m%[0m [0mfold[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m         [0mmonitor[0m[0;34m=[0m[0;34m"val_loss"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py[0m in [0;36m__init__[0;34m(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)[0m
[1;32m    182[0m         [0;32mif[0m [0msave_weights_only[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mfilepath[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m".weights.h5"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 184[0;31m                 raise ValueError(
[0m[1;32m    185[0m                     [0;34m"When using `save_weights_only=True` in `ModelCheckpoint`"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    186[0m                     [0;34m", the filepath provided must end in `.weights.h5` "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=fold-0.h5

## === cell 11
if TEST:
    if PREDICT_SLOPE:
        predictions = np.mean(predictions,axis = 0)
        for i in range(1,len(test_data)+1):
            submission.loc[i,"FVC"] = test_data.loc[i-1,"FVC"] + predictions[i-1,0]*test_data.loc[i-1,"Weekdiff_target"]
            submission.loc[i, "Confidence"] = abs(predictions[i-1,1]*test_data.loc[i-1,"Weekdiff_target"])
    else:
        predictions = np.abs(predictions)
        predictions[:,:,1] = np.power(predictions[:,:,1],2)
        predictions = np.mean(predictions, axis = 0)
        predictions[:,1] = np.power(predictions[:,1],0.5)
        for i in range(1,len(test_data)+1):
            submission.loc[i,"FVC"] = predictions[i-1,0]
            submission.loc[i, "Confidence"] = predictions[i-1,1]
    submission.to_csv("submission.csv", index = False)
