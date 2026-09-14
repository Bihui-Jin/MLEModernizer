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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8496524629797522

# 6. Current score

0.23281

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.74963) has done: 'The fix adds a small protobuf compatibility patch before importing TensorFlow to stop the `MessageFactory` attribute error, and updates the test‑image loading loop to skip sub‑directories so only image files are processed. These changes let the script run end‑to‑end and correctly write a `submission.csv` file.'
- What this solution (achieved 0.7571) has done: 'I increase the number of training epochs for the classifier (from 5 to 15) so the model can learn more from the extracted EfficientNet features, which should raise validation accuracy and move the public score closer to the target while keeping the overall architecture unchanged.'
- What this solution (achieved 0.23281) has done: 'The changes add a small hidden dense layer and increase the training epochs (while keeping early‑stopping) so the classifier can learn richer representations from the EfficientNet features, which should raise validation accuracy and move the public score closer to the target. All other logic, data handling and submission generation remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        message_factory.MessageFactory.GetPrototype = (
            lambda self, descriptor: self.GetMessageClass(descriptor)
        )
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)
np.random.seed(42)




## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)




## === cell 3
model_path = "/kaggle/input/finalsubmit/myfinal_model555.h5"
custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
else:
    train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    train_df["label"] = train_df["label"].astype(str)

    base_extractor = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_extractor.trainable = False

    BATCH_SIZE_FEAT = 256
    train_gen_feat = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    ).flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(WORK_DIR, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=BATCH_SIZE_FEAT,
        class_mode=None,
        shuffle=False,
        workers=4,
        use_multiprocessing=True,
    )

    steps_feat = int(np.ceil(train_gen_feat.samples / BATCH_SIZE_FEAT))
    train_features = base_extractor.predict(
        train_gen_feat, steps=steps_feat, verbose=1
    )  # (num_samples, 1280)

    y_int = train_df["label"].astype(int).values
    train_labels = to_categorical(y_int, num_classes=5)

    classifier = Sequential(
        [
            Dropout(0.2, input_shape=(train_features.shape[1],)),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(5, activation="softmax"),
        ]
    )
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy")
    rl = ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_accuracy")

    classifier.fit(
        train_features,
        train_labels,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        callbacks=[es, rl],
        verbose=2,
    )

    model = Sequential(
        [
            base_extractor,
            Dropout(0.2),
            Dense(5, activation="softmax"),
        ]
    )
    model.layers[-1].set_weights(classifier.layers[-1].get_weights())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3283272270.py in <cell line: 0>()
     76     )
     77     # Transfer only the final dense layer weights (the hidden layer is not part of the final model)
---> 78     model.layers[-1].set_weights(classifier.layers[-1].get_weights())
     79 
     80 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in set_weights(self, weights)
    704         for variable, value in zip(layer_weights, weights):
    705             if variable.shape != value.shape:
--> 706                 raise ValueError(
    707                     f"Layer {self.name} weight shape {variable.shape} "
    708                     "is not compatible with provided weight "

ValueError: Layer dense_2 weight shape (1280, 5) is not compatible with provided weight shape (256, 5).

## === cell 4
test_dir = os.path.join(WORK_DIR, "test_images")
image_names = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

batch_size = 32
pred_labels = []

for i in range(0, len(image_names), batch_size):
    batch_names = image_names[i : i + batch_size]
    batch_images = np.empty((len(batch_names), 224, 224, 3), dtype=np.float32)
    for idx, name in enumerate(batch_names):
        img_path = os.path.join(test_dir, name)
        img = kimage.load_img(img_path, target_size=(224, 224))
        batch_images[idx] = kimage.img_to_array(img)
    x_batch = preprocess_input(batch_images)
    preds = model.predict(x_batch, verbose=0)
    batch_preds = np.argmax(preds, axis=-1).astype(int)
    pred_labels.extend(batch_preds.tolist())

submission = pd.DataFrame({"image_id": image_names, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
