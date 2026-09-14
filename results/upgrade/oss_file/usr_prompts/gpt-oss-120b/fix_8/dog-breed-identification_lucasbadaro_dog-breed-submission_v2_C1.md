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

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        input/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
            test/
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
            train/
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> working/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> working/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

# 5. Target score

0.28939

# 6. Current score

0.49046

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.73077) has done: 'I replace the problematic `keras` imports with the TensorFlow Keras API, fix the shape error by computing features inside `RedeNeural.predict`, and ensure the pipeline writes a correctly‑formatted CSV whose rows contain valid probability vectors.'
- What this solution (achieved 0.44602) has done: 'I fixed the import errors caused by a protobuf‑TensorFlow clash by switching all Keras imports to the standalone `keras` package and removed the unnecessary image‑scaling step that was hurting the model’s log‑loss. The pipeline now only reshapes the raw uint8 images before feature extraction, keeping the pixel values in the expected range for InceptionResNetV2 preprocessing. These minimal changes resolve the runtime exception and should lower the log‑loss toward the target score while preserving the original model architecture.'
- What this solution (achieved 0.48231) has done: 'I import the missing sklearn classes, guard the TensorFlow import to avoid the protobuf error, and ensure the pipeline variable is created correctly so the later cells can run. The probability predictions already use a soft‑max layer, so they sum to 1 and the submission file be written in the required format.'
- What this solution (achieved 0.49046) has done: 'I replaced the failing TensorFlow import with a pure Keras import and aliased the needed Keras modules so the rest of the code can keep using the same names (`layers`, `Model`, `Input`, `Sequential`, utils, etc.). All `tf.keras` calls were switched to their Keras equivalents, fixing the protobuf‑related import error and ensuring the image‑loading and one‑hot utilities work. I also increased the training epochs from 20 to 30 to give the model a modest boost toward the target log‑loss while preserving the original architecture.'

# 9. Code solution

## === cell 0
import os, gc
import numpy as np, pandas as pd

import keras
from keras import layers, Model, Input, Sequential
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.utils import load_img, img_to_array, to_categorical

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

label = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
breeds = np.unique(label["breed"]).astype(str)




## === cell 2
IMG_H, IMG_W = 299, 299

num_train = label.shape[0]
x_train = np.zeros([num_train, IMG_H, IMG_W, 3], dtype=np.uint8)
train_dir = "/kaggle/input/dog-breed-identification/train"

for i, fname in enumerate(label["id"].values):
    path = os.path.join(train_dir, f"{fname}.jpg")
    img = load_img(path, target_size=(IMG_H, IMG_W))
    x_train[i] = img_to_array(img)

y_train = label["breed"].values




## === cell 3
def get_features(X):
    """
    Build a temporary InceptionResNetV2 feature extractor,
    apply the Keras preprocessing, and return pooled features.
    """
    inputs = Input(shape=X.shape[1:])
    x = preprocess_input(inputs)
    conv_base = InceptionResNetV2(weights="imagenet", include_top=False)
    x = conv_base(x)
    x = layers.GlobalAveragePooling2D()(x)
    model = Model(inputs=inputs, outputs=x)
    return model.predict(X, verbose=0)




## === cell 4
class RedeNeural(BaseEstimator, ClassifierMixin):
    def __init__(
        self, epochs=30, batch_size=128
    ):  # increased epochs for better performance
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        self.labels, ids = np.unique(y, return_inverse=True)
        y_hot = to_categorical(ids)
        n_classes = self.labels.shape[0]

        features = get_features(X)

        self.model = Sequential(
            [
                layers.Dense(256, input_shape=(features.shape[1],)),
                layers.Dropout(0.5),
                layers.Dense(n_classes, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(
            features,
            y_hot,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=2,
        )
        return self

    def predict(self, X, y=None, to_submission=False):
        features = get_features(X)
        probabilities = self.model.predict(features, verbose=0)
        if not to_submission:
            preds = self.labels[np.argmax(probabilities, axis=1)]
            return preds
        return probabilities


class MudaShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.reshape((-1, IMG_H, IMG_W, 3))


pipeline = Pipeline(
    [
        ("reshape", MudaShape()),
        ("ann", RedeNeural(epochs=30)),
    ]
)




## === cell 5
pipeline.fit(x_train, y_train)




## === cell 6
del x_train, y_train, label
gc.collect()




## === cell 7
test_dir = "/kaggle/input/dog-breed-identification/test"
test_filenames = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(test_dir, f))
]

num_test = len(test_filenames)
x_test = np.zeros([num_test, IMG_H, IMG_W, 3], dtype=np.uint8)

for i, fname in enumerate(test_filenames):
    img = load_img(fname, target_size=(IMG_H, IMG_W))
    x_test[i] = img_to_array(img)




## === cell 8
preds_df = pd.DataFrame(columns=np.concatenate((["id"], breeds)))




## === cell 9
preds_df["id"] = [os.path.basename(p).split(".")[0] for p in test_filenames]




## === cell 10
x_test_reshaped = pipeline.named_steps["reshape"].transform(x_test)
y_pred = pipeline.named_steps["ann"].predict(x_test_reshaped, to_submission=True)




## === cell 11
assert y_pred.shape[0] == len(preds_df), "Prediction length mismatch"
preds_df[breeds] = y_pred




## === cell 12
preds_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
