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

0.28939

# 6. Current score

0.38959

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38786) has done: 'I fix the environment import crash by forcing Keras to use the TensorFlow backend early (this avoids the protobuf `MessageFactory.GetPrototype` issue seen in some Kaggle images). I also fix test file discovery to ignore the nested `test/` directory that caused `IsADirectoryError`, and I ensure predictions are generated for exactly the same ids (and in the same order) used to build the submission dataframe. Finally, I make `RedeNeural.predict(to_submission=True)` output a full 120-class probability table aligned to the submission columns (not just the classes seen during training), so the CSV has the correct shape and header for Kaggle.'
- What this solution (achieved 0.38927) has done: 'I fix the import-time crash (`MessageFactory` / protobuf incompatibility) by forcing TensorFlow to use the pure-Python protobuf implementation before importing Keras/TensorFlow, which is the most reliable Kaggle-side workaround. Then I fix a silent but score-harming logic bug: during training you extract InceptionResNetV2 features, but during inference your `RedeNeural.predict()` was accidentally predicting on raw features with the wrong interface—so I make `RedeNeural.predict()` always extract features internally (consistent with `fit()`), keeping the same model/loss/training loop. Finally, I keep the submission aligned to `sample_submission.csv` columns and ids, writing a valid `submission.csv`.'
- What this solution (achieved 0.40041) has done: 'You’re hitting the protobuf `MessageFactory.GetPrototype` crash because importing `keras` (Keras 3) is still pulling in an incompatible protobuf/TensorFlow stack in this Kaggle image despite the env vars. The smallest reliable fix is to stop using Keras 3 here and instead use `tf_keras` (TensorFlow Keras) for `applications`, `layers`, image loading, and `to_categorical`, which avoids the protobuf incompatibility while keeping the exact same model, loss, and training loop. I also remove the dead/incorrect reshape transformer (it would break shapes if ever used) and keep the submission alignment exactly as required (sample_submission columns, test ids order). This should run end-to-end and, because it restores stable feature extraction/training without import crashes, it should move logloss down toward your 0.28939 target from 0.38927 without changing the core approach.'
- What this solution (achieved 0.41594) has done: 'We fix the import-time protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation and (crucially) importing `tensorflow` before `tf_keras`, which avoids the `MessageFactory.GetPrototype` issue in this environment. Then we make feature extraction use a single cached InceptionResNetV2 backbone (instead of rebuilding it every call), which preserves the exact same model/feature logic but removes repeated reinitialization overhead and instability, typically improving training quality within the same epoch budget. Finally, we ensure deterministic file discovery and keep the submission strictly aligned to `sample_submission.csv` columns and test ids, producing a valid `submission.csv`.'
- What this solution (achieved 0.39756) has done: 'We fix the runtime import crash (`MessageFactory.GetPrototype`) by switching from the fragile protobuf-env-var workaround to an in-process monkeypatch that restores `GetPrototype` in older protobuf builds before TensorFlow/tf_keras are imported. This keeps your exact modeling/training/prediction logic intact, but unblocks the pipeline end-to-end so it can train and write `submission.csv`. While touching the feature extractor, we also ensure it’s built once and used consistently (as you intended) without changing architecture or preprocessing. No score-tuning changes are introduced beyond restoring correct execution; this should move you back from “no run” to a valid submission and likely back toward your prior achieved range.'
- What this solution (achieved 4.72078) has done: 'You’re currently training on raw uint8 pixels without the intended `/255` scaling because the `DividePor255` transformer is defined but never used in the pipeline; adding it back is a minimal, core-logic-preserving change that typically reduces logloss for this setup. To keep evaluation semantics identical and avoid silent misalignment, I also force the `test_ids` ordering to match `sample_submission.csv` exactly (instead of relying on filesystem sorting). Finally, I keep the same model/feature extractor/training loop, and only make small determinism/memory safety tweaks that don’t change the approach but help stability and reproducibility of the run.'
- What this solution (achieved 0.4048) has done: 'Your current 4.72078 logloss is far from the 0.28939 target (lower is better), so we need a small but meaningful fix that improves correctness without changing the core approach (InceptionResNetV2 feature extraction + small dense head + categorical crossentropy). The main score-killer is that your `DividePor255()` scaling is applied before `preprocess_input()`, but `inception_resnet_v2.preprocess_input` expects raw `0..255` RGB and internally scales to `[-1, 1]`; feeding already-divided inputs makes the backbone see near-constant values and yields near-uniform predictions, driving logloss very high. I keep the same architecture and training loop, but remove the `/255` scaling from the pipeline (identity scale) so preprocessing matches the pretrained backbone’s expectations. I also make `get_features()` run with `verbose=0` to reduce overhead and stay well within the time limit (no semantic change).'
- What this solution (achieved 0.39687) has done: 'We make one minimal change that should reliably improve multi-class logloss without altering your core model/feature-extraction/training approach: increase the numeric stability floor used when expanding probabilities to all 120 breeds for submission. Right now you use `1e-15`, which is unnecessarily tiny and can lead to very large `-log(p)` penalties when the true class is one your model never learned well; raising this floor slightly (still extremely small) typically reduces logloss by preventing catastrophic penalties while keeping predictions effectively unchanged elsewhere. This preserves the same feature extractor (InceptionResNetV2 + preprocess_input), the same dense head, the same optimizer/loss, and the same training loop. The rest of the pipeline, test id alignment, and submission formatting remain unchanged.'
- What this solution (achieved 0.38959) has done: 'You’re currently worse than the target (0.39687 vs 0.28939, lower is better), so we should make a small change that legitimately improves logloss without changing the model/feature-extraction/training loop. The biggest lever available inside your existing logic is the probability “floor” you inject for breeds not well covered by the model; `1e-6` is large enough to noticeably distort calibrated softmax outputs and hurt logloss, so I reduce it to a much smaller (still safe) value and also clip/renormalize after expansion to keep rows valid probabilities. This keeps the exact same architecture, loss, optimizer, epochs, preprocessing, and submission alignment, but should reduce logloss by avoiding excessive mass assigned to the wrong classes while still preventing catastrophic `log(0)`. The script still runs end-to-end and writes a valid `submission.csv` with the exact sample submission columns/order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd

import tensorflow as tf  # must come before tf_keras in this environment
import tf_keras as keras
from tf_keras import layers, applications
from tf_keras.utils import to_categorical

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

np.random.seed(42)
tf.random.set_seed(42)

label = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")



## === cell 1
breeds = np.unique(label["breed"]).astype(str)



## === cell 2
x_train = np.zeros([label.shape[0], 331, 331, 3], dtype=np.uint8)
y_train = np.empty([label.shape[0], 1], dtype=breeds.dtype)

for i, filename in enumerate(label["id"].values):
    path = (
        os.path.join("/kaggle/input/dog-breed-identification/train", filename) + ".jpg"
    )
    image = keras.utils.load_img(path, target_size=x_train.shape[1:3])
    input_arr = keras.utils.img_to_array(image)
    x_train[i] = input_arr
    del input_arr
    y_train[i] = label["breed"][i]



## === cell 3
_FEATURE_EXTRACTOR = None


def _get_feature_extractor(input_shape):
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        inputs = keras.Input(shape=input_shape)
        conv_base = applications.inception_resnet_v2.InceptionResNetV2(
            weights="imagenet",
            include_top=False,
        )
        x = applications.inception_resnet_v2.preprocess_input(inputs)
        x = conv_base(x)
        x = layers.GlobalAveragePooling2D()(x)
        _FEATURE_EXTRACTOR = keras.Model(inputs=inputs, outputs=x)
    return _FEATURE_EXTRACTOR


def get_features(X):
    X = np.asarray(X)
    model = _get_feature_extractor(X.shape[1:])
    return model.predict(X, verbose=0)




## === cell 4
class RedeNeural(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=5, batch_size=128, all_classes=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.all_classes = (
            None if all_classes is None else np.array(all_classes, dtype=str)
        )

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self.labels, ids = np.unique(y, return_inverse=True)
        yhot = to_categorical(ids)
        n_classes = self.labels.shape[0]

        features_preprocessn = get_features(X)
        self.feature_dim_ = int(features_preprocessn.shape[1])

        self.model = keras.Sequential(
            [
                layers.Dense(256, input_shape=(self.feature_dim_,)),
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
            features_preprocessn,
            yhot,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1,
        )
        return self

    def predict(self, X, y=None, to_submission=False):
        X = np.asarray(X)
        if X.ndim == 4:
            X_feat = get_features(X)
        elif X.ndim == 2:
            X_feat = X
        else:
            raise ValueError(f"Unexpected X shape for predict: {X.shape}")

        probabilities = self.model.predict(X_feat, verbose=0)

        if not to_submission:
            ypred = self.labels[np.argmax(probabilities, axis=1)]
            return ypred

        if self.all_classes is None:
            return probabilities

        n = probabilities.shape[0]
        floor = 1e-12  # small stability floor; reduces distortion vs 1e-6
        full = np.full((n, len(self.all_classes)), floor, dtype=np.float64)

        class_to_idx = {c: i for i, c in enumerate(self.all_classes)}
        trained_positions = [class_to_idx[c] for c in self.labels]
        full[:, trained_positions] = probabilities.astype(np.float64)

        full = np.clip(full, floor, 1.0)
        full /= full.sum(axis=1, keepdims=True)
        return full.astype(np.float32)


class DividePor255(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=np.uint8)


class MudaShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


modelo = Pipeline(
    [
        ("scale", DividePor255()),
        ("ann", RedeNeural(epochs=20, all_classes=breeds)),
    ]
)



## === cell 5
modelo.fit(x_train, y_train)



## === cell 6
del x_train
del y_train
del label



## === cell 7
sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
test_ids = sample_sub["id"].astype(str).tolist()

test_dir = "/kaggle/input/dog-breed-identification/test"
test_paths = [os.path.join(test_dir, tid + ".jpg") for tid in test_ids]

x_test = np.zeros([len(test_paths), 331, 331, 3], dtype=np.uint8)
for i, path in enumerate(test_paths):
    image = keras.utils.load_img(path, target_size=x_test.shape[1:3])
    input_arr = keras.utils.img_to_array(image)
    x_test[i] = input_arr
    del input_arr



## === cell 8
y_pred = modelo.predict(x_test, to_submission=True)



## === cell 9
del x_test
del test_paths



## === cell 10
import gc

gc.collect()



## === cell 11
preds_df = sample_sub.copy()
preds_df["id"] = test_ids



## === cell 12
breed_cols = [c for c in preds_df.columns if c != "id"]
if y_pred.shape[1] != len(breed_cols):
    raise ValueError(
        f"Prediction shape {y_pred.shape} does not match submission cols ({len(breed_cols)})."
    )

preds_df.loc[:, breed_cols] = y_pred



## === cell 13
preds_df.to_csv("submission.csv", index=False)
preds_df.head()
