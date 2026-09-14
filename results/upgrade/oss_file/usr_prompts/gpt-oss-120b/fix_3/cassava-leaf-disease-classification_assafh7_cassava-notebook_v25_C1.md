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

0.853732245391357

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

dataset_root = "/kaggle/input"
print("Dataset root exists:", os.path.isdir(dataset_root))




## === cell 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

try:
    import albumentations as aug
except Exception as e:
    print("Albumentations could not be imported:", e)
    aug = None




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
warnings.simplefilter("ignore")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2099231708.py in <cell line: 0>()
----> 1 warnings.simplefilter("ignore")
      2 
      3 

NameError: name 'warnings' is not defined

## === cell 3
general_path = "/kaggle/input/cassava-leaf-disease-classification/"
print("Listing root of dataset:")
print(os.listdir(general_path))




## === cell 4
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.load(file)
    map_classes = {int(k): v for k, v in map_classes.items()}
print(json.dumps(map_classes, indent=4))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3003526786.py in <cell line: 0>()
      1 with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
----> 2     map_classes = json.load(file)
      3     map_classes = {int(k): v for k, v in map_classes.items()}
      4 print(json.dumps(map_classes, indent=4))
      5 

NameError: name 'json' is not defined

## === cell 5
train_images_path = os.path.join(general_path, "train_images")
train_image_files = os.listdir(train_images_path)
print(f"Number of train images: {len(train_image_files)}")




## === cell 6
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2526174577.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
      2 df_train["class_name"] = df_train["label"].map(map_classes)
      3 df_train.head()
      4 
      5 

NameError: name 'pd' is not defined

## === cell 7
plt.figure(figsize=(8, 4))
sns.countplot(y="class_name", data=df_train)
plt.title("Class distribution")
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/975099261.py in <cell line: 0>()
----> 1 plt.figure(figsize=(8, 4))
      2 sns.countplot(y="class_name", data=df_train)
      3 plt.title("Class distribution")
      4 plt.show()
      5 

NameError: name 'plt' is not defined

## === cell 8
def visualize_batch(image_ids, labels, class_names):
    plt.figure(figsize=(16, 12))
    for ind, (image_id, label, cname) in enumerate(zip(image_ids, labels, class_names)):
        plt.subplot(3, 3, ind + 1)
        img = cv2.imread(os.path.join(train_images_path, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Class {label}: {cname}", fontsize=12)
        plt.axis("off")
    plt.show()




## === cell 9
tmp = df_train[df_train["label"] == 0].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3127729601.py in <cell line: 0>()
----> 1 tmp = df_train[df_train["label"] == 0].sample(6, random_state=42)
      2 visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)
      3 
      4 

NameError: name 'df_train' is not defined

## === cell 10
tmp = df_train[df_train["label"] == 1].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2495671649.py in <cell line: 0>()
----> 1 tmp = df_train[df_train["label"] == 1].sample(6, random_state=42)
      2 visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)
      3 
      4 

NameError: name 'df_train' is not defined

## === cell 11
tmp = df_train[df_train["label"] == 2].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3639671656.py in <cell line: 0>()
----> 1 tmp = df_train[df_train["label"] == 2].sample(6, random_state=42)
      2 visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)
      3 
      4 

NameError: name 'df_train' is not defined

## === cell 12
tmp = df_train[df_train["label"] == 3].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3558045008.py in <cell line: 0>()
----> 1 tmp = df_train[df_train["label"] == 3].sample(6, random_state=42)
      2 visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)
      3 
      4 

NameError: name 'df_train' is not defined

## === cell 13
tmp = df_train[df_train["label"] == 4].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1993635604.py in <cell line: 0>()
----> 1 tmp = df_train[df_train["label"] == 4].sample(6, random_state=42)
      2 visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)
      3 
      4 

NameError: name 'df_train' is not defined

## === cell 14
if aug is not None:
    transform_shift_scale_rotate = aug.ShiftScaleRotate(
        p=1.0,
        shift_limit=(-0.3, 0.3),
        scale_limit=(-0.1, 0.1),
        rotate_limit=(-180, 180),
        interpolation=0,
        border_mode=4,
    )

    def plot_augmentation(image_id, transform):
        plt.figure(figsize=(12, 12))
        img = cv2.imread(os.path.join(train_images_path, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("original")
        for i in range(2, 5):
            aug_img = transform(image=img)["image"]
            plt.subplot(2, 2, i)
            plt.imshow(aug_img)
            plt.axis("off")
            plt.title(f"aug {i-1}")
        plt.show()

    plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1235701880.py in <cell line: 0>()
     25         plt.show()
     26 
---> 27     plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)
     28 
     29 

/tmp/ipykernel_12/1235701880.py in plot_augmentation(image_id, transform)
     10 
     11     def plot_augmentation(image_id, transform):
---> 12         plt.figure(figsize=(12, 12))
     13         img = cv2.imread(os.path.join(train_images_path, image_id))
     14         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

NameError: name 'plt' is not defined

## === cell 15
img_width, img_height = 260, 260




## === cell 16
train_df = pd.read_csv(os.path.join(general_path, "train.csv"))
train_df["label"] = train_df["label"].astype(str)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3118057783.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(os.path.join(general_path, "train.csv"))
      2 train_df["label"] = train_df["label"].astype(str)
      3 
      4 

NameError: name 'pd' is not defined

## === cell 17
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0 / 255,
)

train_flow = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_images_path,
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

valid_flow = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_images_path,
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1272930614.py in <cell line: 0>()
      9 
     10 train_flow = train_datagen.flow_from_dataframe(
---> 11     dataframe=train_df,
     12     directory=train_images_path,
     13     x_col="image_id",

NameError: name 'train_df' is not defined

## === cell 18
x_batch, y_batch = next(train_flow)
print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3836726193.py in <cell line: 0>()
----> 1 x_batch, y_batch = next(train_flow)
      2 print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
      3 
      4 

NameError: name 'train_flow' is not defined

## === cell 19
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(5, activation="softmax")(x)  # 5 classes
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 20
history = model.fit(
    train_flow,
    epochs=3,
    validation_data=valid_flow,
    verbose=1,
    workers=4,
    use_multiprocessing=True,
)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/595905424.py in <cell line: 0>()
      1 # Added multiprocessing workers to speed up data loading and augmentation.
      2 history = model.fit(
----> 3     train_flow,
      4     epochs=3,
      5     validation_data=valid_flow,

NameError: name 'train_flow' is not defined

## === cell 21
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_flow,
    epochs=2,
    validation_data=valid_flow,
    verbose=1,
    workers=4,
    use_multiprocessing=True,
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1605759220.py in <cell line: 0>()
     10 # Added workers here as well.
     11 model.fit(
---> 12     train_flow,
     13     epochs=2,
     14     validation_data=valid_flow,

NameError: name 'train_flow' is not defined

## === cell 22
sample_sub = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
sample_sub.head()




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2457675091.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
      2 sample_sub.head()
      3 
      4 

NameError: name 'pd' is not defined

## === cell 23
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_flow = test_datagen.flow_from_dataframe(
    dataframe=sample_sub,
    directory=os.path.join(general_path, "test_images"),
    x_col="image_id",
    y_col=None,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode=None,
    shuffle=False,
)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1349381977.py in <cell line: 0>()
      2 
      3 test_flow = test_datagen.flow_from_dataframe(
----> 4     dataframe=sample_sub,
      5     directory=os.path.join(general_path, "test_images"),
      6     x_col="image_id",

NameError: name 'sample_sub' is not defined

## === cell 24
test_predictions = model.predict(test_flow, steps=len(test_flow), verbose=1)
test_labels = np.argmax(test_predictions, axis=1)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/717936623.py in <cell line: 0>()
----> 1 test_predictions = model.predict(test_flow, steps=len(test_flow), verbose=1)
      2 test_labels = np.argmax(test_predictions, axis=1)
      3 
      4 

NameError: name 'test_flow' is not defined

## === cell 25
submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": test_labels})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
print(submission.head())

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/104945334.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": test_labels})
      2 submission.to_csv("submission.csv", index=False)
      3 print("Submission file written to submission.csv")
      4 print(submission.head())

NameError: name 'pd' is not defined
