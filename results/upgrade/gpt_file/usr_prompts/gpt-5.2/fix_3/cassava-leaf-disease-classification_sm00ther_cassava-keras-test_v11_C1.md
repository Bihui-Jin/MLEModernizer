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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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

0.6128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import sys, subprocess


def _ensure_protobuf_compatible():
    """
    BUGFIX: Do not try to downgrade protobuf in this Kaggle environment.
    - TensorFlow 2.18 expects protobuf>=5; forcing protobuf<5 can break TF imports/runtime.
    - The original notebook attempted to downgrade protobuf which is incompatible here.
    Keeping as a no-op is score-neutral and stabilizes execution.
    """
    return


_ensure_protobuf_compatible()

import tensorflow as tf
import albumentations as A

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAvgPool2D,
    ELU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import CategoricalAccuracy

from sklearn.model_selection import StratifiedKFold, train_test_split

np.random.seed(42)
tf.random.set_seed(42)

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("Top files:", os.listdir(WORK_DIR)[:10])

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json"), "r") as file:
    labels = json.load(file)
labels



## === cell 2
data = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_images_folder = os.path.join(WORK_DIR, "train_images")
test_images_folder = os.path.join(WORK_DIR, "test_images")

rand_idx = np.random.randint(data.shape[0])
rand_fname = data.iloc[rand_idx, 0]
rand_fpath = os.path.join(train_images_folder, rand_fname)
print("Random image:", rand_fpath)
Image.open(rand_fpath)




## === cell 3
class MyGeneratorUnWeighted(tf.keras.utils.Sequence):
    def __init__(
        self,
        x,
        y,
        batch_size,
        img_size,
        num_classes,
        augment_img,
        imgs_folder,
        img_preproc=False,
    ):
        self.x, self.y = list(x), list(y)
        self.img_height, self.img_width = img_size
        self.batch_size = batch_size
        self.img_preproc = img_preproc
        self.augment_img = augment_img
        self.num_classes = num_classes
        self.imgs_folder = imgs_folder

        self._transform = A.Compose(
            [
                A.RandomCrop(
                    p=0.5,
                    width=int(self.img_width / 1.3),
                    height=int(self.img_height / 1.3),
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    p=0.2, brightness_limit=0.1, contrast_limit=0.1
                ),
                A.ShiftScaleRotate(p=0.5),
                A.RandomRotate90(),
                A.Blur(p=0.2, blur_limit=2),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, border_mode=4),
                A.GridDistortion(distort_limit=0.2, border_mode=4),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=40
                ),
                A.CLAHE(),
                A.ImageCompression(quality_lower=85),
            ]
        )

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def _augment(self, img):
        return self._transform(image=img)["image"]

    def preproc_imgs(self, img):
        img = img / 255.0
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - imagenet_mean) / imagenet_std
        return img

    def __getitem__(self, idx):
        batch_x = np.zeros(
            (self.batch_size, self.img_height, self.img_width, 3), dtype=np.float32
        )
        batch_y = np.zeros(self.batch_size, dtype=np.int32)

        x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        for n, (i, l) in enumerate(zip(x, y)):
            img = load_img(os.path.join(self.imgs_folder, i))
            img = img_to_array(img, dtype="uint8")

            if self.augment_img:
                img = self._augment(img)

            img = cv2.resize(img, (self.img_width, self.img_height))

            if self.img_preproc:
                img = self.preproc_imgs(img)
            else:
                img = img.astype(np.float32) / 255.0

            batch_x[n] = img
            batch_y[n] = int(l)

        return batch_x, to_categorical(batch_y, num_classes=self.num_classes)




## === cell 4
x_train, x_test, y_train, y_test = train_test_split(
    data.image_id, data.label, test_size=0.25, stratify=data.label, random_state=42
)
len(x_train), len(x_test), np.bincount(y_train)



## === cell 5
plt.figure(figsize=(16, 18))
check_gen = MyGeneratorUnWeighted(
    x_train,
    y_train,
    batch_size=1,
    img_preproc=True,
    augment_img=True,
    imgs_folder=train_images_folder,
    num_classes=5,
    img_size=(384, 384),
)
for n in range(6):
    plt.subplot(3, 2, n + 1)
    check_img = check_gen.__getitem__(0)[0][0]
    img_disp = (check_img - check_img.min()) / (
        check_img.max() - check_img.min() + 1e-6
    )
    plt.imshow(img_disp)
    plt.axis("off")
plt.show()



## === cell 6
net_cfg = {
    "num_classes": 5,
    "img_height": 384,
    "img_width": 384,
    "img_channels": 3,
    "batch_size": 16,
    "backbone": EfficientNetB3(include_top=False, weights=None),
    "preproc_img": True,
    "augment_img": True,
    "num_epochs": 1,
    "class_weights": None,
    "num_folds": 2,
    "backbone_trainable": True,
    "use_dropout": True,
    "activation": ELU(),
    "opt": Adam(learning_rate=0.0001),
    "criterion": CategoricalCrossentropy(),
    "metrics": [CategoricalAccuracy()],
}
net_cfg




## === cell 7
class CassavaNet:
    def __init__(self, model_name, net_cfg, x_train, y_train, x_test, y_test):
        self.model_name = model_name
        self.num_classes = net_cfg["num_classes"]
        self.batch_size = net_cfg["batch_size"]
        self.img_height = net_cfg["img_height"]
        self.img_width = net_cfg["img_width"]
        self.img_channels = net_cfg["img_channels"]
        self.preproc_img = net_cfg["preproc_img"]
        self.augment_img = net_cfg["augment_img"]
        self.backbone = net_cfg["backbone"]
        self.backbone.trainable = net_cfg["backbone_trainable"]

        inputs = Input(shape=(self.img_height, self.img_width, self.img_channels))
        x = BatchNormalization()(inputs)
        x = self.backbone(x)
        x = GlobalAvgPool2D()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = net_cfg["activation"](x)

        if net_cfg["use_dropout"]:
            x = Dropout(0.05)(x)

        outputs = Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs, outputs)

        self.opt = net_cfg["opt"]
        self.criterion = net_cfg["criterion"]
        self.metrics = net_cfg["metrics"]
        self.class_weights = net_cfg["class_weights"]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.weights_path = f"{self.model_name}.weights.h5"

        cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.weights_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
        cb_earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, verbose=0, restore_best_weights=True
        )
        self.callbacks = [cb_tensorboard, cb_checkpoint, cb_earlystop]

        self.traingen = MyGeneratorUnWeighted(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            img_preproc=self.preproc_img,
            augment_img=self.augment_img,
            imgs_folder=train_images_folder,
            num_classes=self.num_classes,
            img_size=(self.img_height, self.img_width),
        )
        self.testgen = MyGeneratorUnWeighted(
            self.x_test,
            self.y_test,
            batch_size=self.batch_size,
            img_preproc=self.preproc_img,
            augment_img=False,
            imgs_folder=train_images_folder,
            num_classes=self.num_classes,
            img_size=(self.img_height, self.img_width),
        )

        self.history = None
        self.model.summary()

    def _compile(self):
        self.model.compile(
            optimizer=self.opt, loss=self.criterion, metrics=self.metrics
        )

    def train(self):
        self._compile()

        self.history = self.model.fit(
            self.traingen,
            steps_per_epoch=len(self.traingen),
            epochs=net_cfg["num_epochs"],
            validation_data=self.testgen,
            validation_steps=len(self.testgen),
            callbacks=self.callbacks,
            class_weight=self.class_weights,
            verbose=1,
        )




## === cell 8
skf = StratifiedKFold(n_splits=net_cfg["num_folds"], random_state=42, shuffle=True)

for fold, (train_skf_idx, test_skf_idx) in enumerate(
    skf.split(data.image_id, data.label)
):
    print(f"...Fold # {fold}...")
    x_train_fold = data.iloc[train_skf_idx, 0].tolist()
    y_train_fold = data.iloc[train_skf_idx, 1].tolist()
    x_test_fold = data.iloc[test_skf_idx, 0].tolist()
    y_test_fold = data.iloc[test_skf_idx, 1].tolist()

    cassava_net = CassavaNet(
        "m" + str(fold),
        net_cfg,
        x_train_fold[:500],
        y_train_fold[:500],
        x_test_fold[:500],
        y_test_fold[:500],
    )
    cassava_net.train()
    del cassava_net

print("Saved weight files:", [f for f in os.listdir(".") if f.endswith(".weights.h5")])




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/3570524654.py in <cell line: 0>()
     18         y_test_fold[:500],
     19     )
---> 20     cassava_net.train()
     21     del cassava_net
     22 

/tmp/ipykernel_11/2223053060.py in train(self)
     85         # BUGFIX: TF/Keras 2.18 removed/does not accept `workers` and `use_multiprocessing`
     86         # for the default trainer. Removing them unblocks training and weight saving.
---> 87         self.history = self.model.fit(
     88             self.traingen,
     89             steps_per_epoch=len(self.traingen),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_numpy(x)
    153     elif isinstance(x, tf.RaggedTensor):
    154         x = x.to_tensor()
--> 155     return np.array(x)
    156 
    157 

NotImplementedError: numpy() is only available when eager execution is enabled.

## === cell 9
def preproc_one_image(fname, img_height, img_width):
    img = load_img(fname)
    img = img_to_array(img, dtype="uint8")
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - imagenet_mean) / imagenet_std
    img = np.expand_dims(img, axis=0)
    return img




## === cell 10
def predict_ensemble(num_models, test_imgs_folder):
    sample_sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    fnames = sample_sub["image_id"].tolist()

    models_preds = []

    for m in range(num_models):
        print("Loading model", m)
        models_preds.append([])

        dummy_x = [data.iloc[0, 0]]
        dummy_y = [int(data.iloc[0, 1])]
        dummy_xt = [data.iloc[1, 0]]
        dummy_yt = [int(data.iloc[1, 1])]

        cassava_net = CassavaNet(
            "m" + str(m), net_cfg, dummy_x, dummy_y, dummy_xt, dummy_yt
        )
        weights_path = "m" + str(m) + ".weights.h5"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights: {weights_path}")
        cassava_net.model.load_weights(weights_path)

        for image_id in fnames:
            fpath = os.path.join(test_imgs_folder, image_id)
            img = preproc_one_image(fpath, net_cfg["img_height"], net_cfg["img_width"])
            pred = np.argmax(cassava_net.model.predict(img, verbose=0), axis=1)[0]
            models_preds[m].append(int(pred))

        del cassava_net

    models_preds = np.array(models_preds).T  # (N, num_models)
    final_preds = []
    for i in range(models_preds.shape[0]):
        votes = models_preds[i]
        final_preds.append(
            int(np.argmax(np.bincount(votes, minlength=net_cfg["num_classes"])))
        )

    submission = pd.DataFrame({"image_id": fnames, "label": final_preds})

    submission = submission[["image_id", "label"]]
    submission.to_csv("submission.csv", index=False)
    return submission




## === cell 11
sub = predict_ensemble(net_cfg["num_folds"], test_images_folder)
print(sub.head())
print("submission.csv written:", os.path.exists("submission.csv"), "rows:", len(sub))

with open("submission.csv", "r") as f:
    for _ in range(10):
        print(f.readline().strip())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2081287452.py in <cell line: 0>()
----> 1 sub = predict_ensemble(net_cfg["num_folds"], test_images_folder)
      2 print(sub.head())
      3 print("submission.csv written:", os.path.exists("submission.csv"), "rows:", len(sub))
      4 
      5 with open("submission.csv", "r") as f:

/tmp/ipykernel_11/4134086631.py in predict_ensemble(num_models, test_imgs_folder)
     23         weights_path = "m" + str(m) + ".weights.h5"
     24         if not os.path.exists(weights_path):
---> 25             raise FileNotFoundError(f"Missing weights: {weights_path}")
     26         cassava_net.model.load_weights(weights_path)
     27 

FileNotFoundError: Missing weights: m1.weights.h5
