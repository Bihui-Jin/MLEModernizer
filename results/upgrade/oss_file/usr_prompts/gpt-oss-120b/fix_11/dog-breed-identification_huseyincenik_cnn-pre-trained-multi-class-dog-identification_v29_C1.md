# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

# 3. Installed packages

colorama==0.4.6
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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scikit-plot==0.3.7
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
termcolor==3.1.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.3496

# 6. Current score

0.82954

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.15187) has done: 'I fix the protobuf import issue, add missing sklearn imports, correctly create the label encoder and class‑weight dictionary, simplify the training loop to run quickly, and ensure a proper submission CSV is written with the required columns.'
- What this solution (achieved 0.81766) has done: 'I remove the faulty protobuf monkey‑patch, replace it with a safe dummy `GetPrototype` implementation, and keep the environment variable settings. This restores all required imports (including `load_img`, `LabelEncoder`, `train_test_split`, etc.) so the subsequent cells can run without NameErrors. The dummy `GetPrototype` avoids the protobuf attribute error that occurs when loading the ResNet model. No other logic is changed, preserving the original workflow while enabling a valid CSV submission.'
- What this solution (achieved 1.05879) has done: 'I slightly adjust the model and training to give it a bit more capacity and allow it to train longer, then fine‑tune on the full training set after early‑stopping. This modest change keeps the overall architecture unchanged while expected to lower the log‑loss toward the target.'
- What this solution (achieved 0.97469) has done: 'I adjust the class‑weight calculation to use the full label distribution (instead of only the training split) and add a lightweight temperature‑scaling step that optimizes a single temperature on the validation set to better calibrate the predicted probabilities. This small calibration typically reduces multi‑class log‑loss without altering the model architecture, keeping the core logic intact while moving the score closer to the target.'
- What this solution (achieved 1.46501) has done: 'I fixed the validation‑loss calculation by removing the unnecessary `labels` argument (so `log_loss` works with one‑hot targets) and added a small model‑capacity boost plus a longer training schedule with a higher early‑stopping patience. These changes keep the core workflow intact while correcting the crash and nudging the log‑loss toward the target.'
- What this solution (achieved 1.55531) has done: 'I increase model capacity, add label‑smoothing to the loss, and give the training a bit more room (higher patience and more epochs). These tweaks keep the overall workflow intact while improving calibration and fitting, which should lower the multi‑class log‑loss toward the target.'
- What this solution (achieved 0.98929) has done: 'I remove label‑smoothing from the loss (which can unnecessarily bias the probabilities), skip the extra fine‑tuning on the full dataset (the early‑stopped model already gives the best validation loss), and broaden the temperature‑scaling search to include smaller temperatures so the calibration can better sharpen predictions. These small, targeted tweaks keep the original architecture and workflow while aiming to lower the multi‑class log‑loss toward the target.'
- What this solution (achieved 0.82954) has done: 'The fix corrects the typo that prevented feature extraction for the test set (`ResNet5` → `ResNet50`) and ensures the subsequent cells run in order, producing a valid `submission.csv`. No core logic or model architecture is changed, preserving the original workflow while making the pipeline executable end‑to‑end.'

# 9. Code solution

## === cell 0
import os, random, shutil, csv, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import google.protobuf.message_factory as _msg_factory

    if not hasattr(_msg_factory.MessageFactory, "GetPrototype"):
        _msg_factory.MessageFactory.GetPrototype = lambda self, descriptor: descriptor
except Exception:
    pass  # If protobuf is not available, the rest of the code will raise appropriately

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Lambda,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")



## === cell 1
train_dir = "/kaggle/input/dog-breed-identification/train"
test_dir = "/kaggle/input/dog-breed-identification/test"
labels_path = "/kaggle/input/dog-breed-identification/labels.csv"
sample_submission_path = "/kaggle/input/dog-breed-identification/sample_submission.csv"

labels_df = pd.read_csv(labels_path)
sample_df = pd.read_csv(sample_submission_path)




## === cell 2
def load_images(image_dir, ids, img_size=(224, 224, 3)):
    n = len(ids)
    X = np.zeros((n, img_size[0], img_size[1], img_size[2]), dtype=np.uint8)
    for i, img_id in enumerate(tqdm(ids, desc="Loading images")):
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        img = load_img(img_path, target_size=img_size[:2])
        X[i] = img_to_array(img)
    return X




## === cell 3
train_ids = labels_df["id"].values
X_images = load_images(train_dir, train_ids, img_size=(224, 224, 3))



## === cell 4
le = LabelEncoder()
y_int = le.fit_transform(labels_df["breed"].values)
n_classes = len(le.classes_)
y_onehot = keras.utils.to_categorical(y_int, num_classes=n_classes)



## === cell 5
X_train_img, X_val_img, y_train, y_val = train_test_split(
    X_images, y_onehot, test_size=0.2, stratify=y_int, random_state=42
)




## === cell 6
def get_features(model_cls, preproc, input_shape, data):
    inp = Input(shape=input_shape)
    x = Lambda(preproc)(inp)
    base = model_cls(weights="imagenet", include_top=False, input_tensor=x)
    out = GlobalAveragePooling2D()(base.output)
    extractor = Model(inputs=inp, outputs=out)
    feats = extractor.predict(data, batch_size=64, verbose=0)
    return feats


from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input as resnet_preproc,
)

input_shape = (224, 224, 3)

train_feats = get_features(ResNet50, resnet_preproc, input_shape, X_train_img)
val_feats = get_features(ResNet50, resnet_preproc, input_shape, X_val_img)



## === cell 7
y_train_int_full = y_int  # labels for the entire dataset
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.arange(n_classes), y=y_train_int_full
)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}



## === cell 8
model = Sequential(
    [
        InputLayer(train_feats.shape[1:]),
        Dense(1024, activation="relu"),
        Dropout(0.2),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),  # no label_smoothing
    metrics=[tf.keras.metrics.Recall()],
)

early_stop = EarlyStopping(
    monitor="val_loss", mode="min", patience=50, restore_best_weights=True, verbose=1
)

history = model.fit(
    train_feats,
    y_train,
    validation_data=(val_feats, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=2,
)



## === cell 9
test_ids = sample_df["id"].values
X_test_img = load_images(test_dir, test_ids, img_size=(224, 224, 3))



## === cell 10
test_feats = get_features(
    ResNet50, resnet_preproc, input_shape, X_test_img
)  # corrected model name



## === cell 11
pred_probs = model.predict(test_feats, batch_size=128, verbose=0)




## === cell 12
def apply_temperature(probs, T):
    """Scale softmax probabilities by temperature T."""
    logits = np.log(np.clip(probs, 1e-15, 1.0))
    scaled_logits = logits / T
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


val_preds = model.predict(val_feats, batch_size=128, verbose=0)

temps = np.linspace(0.1, 3.0, 30)
best_T, best_loss = 1.0, np.inf
for T in temps:
    calibrated = apply_temperature(val_preds, T)
    loss = log_loss(y_val, calibrated)
    if loss < best_loss:
        best_loss, best_T = loss, T

fine_temps = np.linspace(max(0.05, best_T - 0.2), best_T + 0.2, 40)
for T in fine_temps:
    calibrated = apply_temperature(val_preds, T)
    loss = log_loss(y_val, calibrated)
    if loss < best_loss:
        best_loss, best_T = loss, T

print(
    f"Optimal temperature: {best_T:.3f} with calibrated val log‑loss: {best_loss:.5f}"
)

pred_probs = apply_temperature(pred_probs, best_T)



## === cell 13
for idx, breed in enumerate(le.classes_):
    sample_df[breed] = pred_probs[:, idx]

submission_path = "submission.csv"
sample_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## === cell 14
sample_df.head()
